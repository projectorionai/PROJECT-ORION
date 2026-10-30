"""Opt-in, bounded swarm lifetime and exclusive, pre-emptible GPU ownership."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from .. import admission
from .lifetime import finish
from .search import SearchProcess, worker_priority
from .training_owner import TrainingLease, external_training_status


def _train_entry(*args):
    worker_priority()
    try:
        from .nnue import train_pass
    except ImportError:
        pipe = args[3]
        pipe.send({"status": "deferred", "reason": "Install the optional Torch training dependencies (CPU or CUDA)"})
        pipe.close()
        return
    train_pass(*args)


def _book_entry(paths, directory, stop, pipe):
    worker_priority()
    try:
        from .book import compile_book
        count = compile_book(paths, Path(directory), stop)
        pipe.send({"status": "book", "entries": count})
    except Exception as exc:  # noqa: BLE001 - bounded worker failure report
        pipe.send({"status": "error", "reason": str(exc)[:180]})
    finally:
        pipe.close()


async def reap(process, stop, *, grace=0.5):
    """Never release a GPU lease until the child is confirmed dead."""
    stop.set()
    await asyncio.to_thread(process.join, grace)
    if process.is_alive():
        await asyncio.to_thread(process.terminate)
        await asyncio.to_thread(process.join, 2)
    if process.is_alive():
        await asyncio.to_thread(process.kill)
        await asyncio.to_thread(process.join)  # safety before capacity, off the UI loop


class Grandmaster:
    def __init__(self, directory: Path, brain_path: Path, bus=None):
        self.directory, self.brain_path, self.bus = Path(directory), Path(brain_path), bus
        self.search = SearchProcess(brain_path, directory)
        self.review = SearchProcess(brain_path, directory)
        self._ctx = mp.get_context("spawn")
        self._stop = self._ctx.Event()
        self._pause = self._ctx.Event()
        self._selfplay = None
        self._tasks: set[asyncio.Task] = set()
        self._owners: set[asyncio.Task] = set()
        self._training_lock = asyncio.Lock()
        self._force = asyncio.Event()
        self._started = False
        self._closed = False
        self._child = None
        self._child_stop = None
        self._loop_name = f"grandmaster-{id(self)}"
        self._live = False
        self._close_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._checkpoint_owner: TrainingLease | None = None

    def emit(self, **payload):
        if self.bus is not None:
            try:
                self.bus.dashboard_event.emit("chess", {"phase": "swarm", **payload})
            except (AttributeError, RuntimeError):
                pass

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        def done(finished):
            self._tasks.discard(finished)
            if not finished.cancelled() and finished.exception() is not None:
                self.emit(status="error", reason=str(finished.exception())[:180])
        task.add_done_callback(done)
        return task

    def foreground(self):
        self._pause.set()
        if self._child_stop is not None:
            self._child_stop.set()

    def resume(self):
        if not self._live:
            self._pause.clear()

    def live_game(self, active: bool):
        self._live = active
        if active:
            self.foreground()
        elif admission.ADMISSION.should_run("training"):
            self.resume()

    async def start(self, pgn_sources=()):
        async with self._lifecycle_lock:
            await self._start(pgn_sources)
            return self.status()

    def _pause_reason(self):
        if self._live:
            return "A live game has priority over training."
        snapshot = admission.ADMISSION.snapshot()
        resources = snapshot.get("resources", {})
        if resources.get("memory") not in {None, "normal"}:
            return f"Memory is {resources['memory']}; training waits for capacity."
        if snapshot.get("foreground_active"):
            return "Foreground work has priority; training resumes when the PC is idle."
        if resources.get("cpu") == "busy" or resources.get("gpu") == "busy":
            return "The processor or graphics card is busy; training waits for capacity."
        return "Waiting for the training scheduler."

    def status(self):
        if not self._started:
            if self._owners:
                return {"status": "running", "reason": "An isolated local practice or evaluation worker is running."}
            external = external_training_status(self.directory)
            if external is not None:
                return external
            return {"status": "stopped", "reason": "Training workers are not running."}
        if self._selfplay is not None and not self._selfplay.is_alive():
            return {"status": "error", "reason": f"Self-play exited ({self._selfplay.exitcode}). Stop and restart the swarm."}
        if self._live or not admission.ADMISSION.should_run("training"):
            return {"status": "paused", "reason": self._pause_reason()}
        return {"status": "running", "reason": "Self-play is collecting complete games for training."}

    async def _start(self, pgn_sources):
        if self._closed:
            raise RuntimeError("Grandmaster closed")
        if self._started:
            return
        from .selfplay import selfplay_main
        owner = await self._acquire_checkpoint_owner()
        if owner is None:
            self.emit(**self.status())
            return
        self._checkpoint_owner = owner
        try:
            admission.ADMISSION.register_loop(self._loop_name, cls="training", pause=self.foreground, resume=self.resume)
            self._selfplay = self._ctx.Process(target=selfplay_main,
                args=(str(self.brain_path), str(self.directory), self._stop, self._pause), daemon=True,
                name="orion-chess-selfplay")
            # Start paused; monitor admits only once the foreground is idle.
            self._pause.set()
            start_task = asyncio.create_task(asyncio.to_thread(self._selfplay.start))
        except BaseException:
            owner.close()
            self._checkpoint_owner = None
            admission.ADMISSION.unregister_loop(self._loop_name)
            raise
        try:
            await asyncio.shield(start_task)
        except BaseException:
            try:
                await finish(start_task)
            finally:
                if self._selfplay.pid is not None:
                    await finish(asyncio.create_task(reap(self._selfplay, self._stop)))
                self._selfplay.close()
                self._selfplay = None
                self._stop, self._pause = self._ctx.Event(), self._ctx.Event()
                admission.ADMISSION.unregister_loop(self._loop_name)
                owner.close()
                self._checkpoint_owner = None
            raise
        self._started = True
        self._spawn(self._monitor())
        self._spawn(self._training_loop())
        if pgn_sources:
            self._spawn(self._book_loop(tuple(pgn_sources)))
        self.emit(status="running")

    async def _monitor(self):
        last_state = None
        while not self._stop.is_set():
            if self._live or not admission.ADMISSION.should_run("training"):
                self._pause.set()
            else:
                self._pause.clear()
            state = self.status()
            if state != last_state:
                self.emit(**state)
                last_state = state
            if self._selfplay is not None and not self._selfplay.is_alive():
                self.emit(status="error", reason=f"Self-play exited ({self._selfplay.exitcode})")
                return
            await asyncio.sleep(0.25)

    async def force_training(self):
        if not self._started:
            await self.start()
        if not self._started:
            # A continuous trainer may be evaluating a candidate already.
            # Do not claim a pass was queued on a supervisor that owns no
            # workers and cannot alter the other trainer's phase.
            state = self.status()
            self.emit(**state)
            return state
        self._force.set()                       # force scheduling, never bypass the VRAM governor
        state = self.status()
        result = {"status": "training queued", "reason": state["reason"]}
        self.emit(**result)
        return result

    async def stop_training(self, *, include_external=True):
        """Stop optional workers without closing the live search processes."""
        async with self._lifecycle_lock:
            if (include_external and not self._started and self._checkpoint_owner is None
                    and not self._owners):
                external = await asyncio.to_thread(external_training_status, self.directory)
                if external is not None:
                    return await self._stop_external(external)
            _result, interrupted = await finish(asyncio.create_task(self._stop_training()))
            if interrupted:
                raise asyncio.CancelledError
            return _result

    async def _stop_external(self, state):
        if state.get("owner") != "continuous":
            self.emit(**state)
            return state
        from ..atomic_io import atomic_write_text
        # The verified CLI watches this sentinel and checkpoints before exit.
        # It remains present so a scheduled worker cannot silently restart.
        await asyncio.to_thread(atomic_write_text, self.directory / "improvement.stop",
                                "Stopped from ORION's training controls.\n", encoding="utf-8")
        for _ in range(40):
            current = await asyncio.to_thread(external_training_status, self.directory)
            if current is None:
                result = {"status": "stopped", "reason": "Continuous training saved its work and stopped. Live chess remains available."}
                self.emit(**result)
                return result
            await asyncio.sleep(0.25)
        result = {"status": "stopping", "reason": "Stop requested. The continuous trainer is still saving or finishing a bounded operation."}
        self.emit(**result)
        return result

    async def _stop_training(self):
        self._stop.set()
        self.foreground()
        admission.ADMISSION.unregister_loop(self._loop_name)
        tasks = tuple((self._tasks | self._owners) - {asyncio.current_task()})
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._selfplay is not None:
            await reap(self._selfplay, self._stop)
            self._selfplay.close()
            self._selfplay = None
        self._started = False
        self._stop = self._ctx.Event()
        self._pause = self._ctx.Event()
        self._force.clear()
        if self._checkpoint_owner is not None:
            self._checkpoint_owner.close()
            self._checkpoint_owner = None
        self.emit(status="stopped")
        return {"status": "stopped", "reason": "Training workers stopped; live chess remains available."}

    async def _run_child(self, target, args, *, gpu_ticket=None, timeout=3600):
        parent, child = self._ctx.Pipe(duplex=False)
        stop = self._ctx.Event()
        process = self._ctx.Process(target=target, args=args(stop, child), daemon=True)
        self._child, self._child_stop = process, stop
        owner = asyncio.current_task()
        already_owned = owner in self._owners
        self._owners.add(owner)
        started = time.monotonic()
        start_task = asyncio.create_task(asyncio.to_thread(process.start))
        try:
            await asyncio.shield(start_task)
            child.close()
            while process.is_alive():
                if gpu_ticket is None:
                    if self._live or not admission.ADMISSION.should_run("training"):
                        self._pause.set()
                    else:
                        self._pause.clear()
                preempt = gpu_ticket is not None and (
                    gpu_ticket.preempt_requested or admission.ADMISSION.foreground_active()
                    or not admission.ADMISSION.should_run("training") or self._live)
                if stop.is_set() or self._stop.is_set() or preempt or time.monotonic() - started > timeout:
                    await reap(process, stop)
                    return {"status": "deferred", "reason": "Pre-empted or time budget reached"}
                await asyncio.sleep(0.025 if gpu_ticket is not None else 0.25)
            await asyncio.to_thread(process.join)
            if parent.poll():
                try:
                    return parent.recv()      # child messages are bounded metadata (<1 KiB)
                except EOFError:
                    pass
            return {"status": "error", "reason": f"Worker exited ({process.exitcode})"}
        finally:
            interrupted = False
            try:
                _result, interrupted = await finish(start_task)
            finally:
                if process.pid is not None:
                    cleanup = asyncio.create_task(reap(process, stop))
                    _result, again = await finish(cleanup)
                    interrupted |= again
                parent.close()
                child.close()
                process.close()
                self._child, self._child_stop = None, None
                if not already_owned:
                    self._owners.discard(owner)
            if interrupted:
                raise asyncio.CancelledError

    async def practice(self, games):
        from .selfplay import practice_main
        async with self._training_lock, self._checkpoint_access() as available:
            if not available:
                return self.status()
            async with admission.ADMISSION.slot("training", label="Chess practice"):
                result = await self._run_child(practice_main,
                    lambda stop, pipe: (str(self.brain_path), str(self.directory), games,
                                        stop, self._pause, pipe), timeout=4 * 3600)
            self.emit(**result)
            return result

    @asynccontextmanager
    async def _checkpoint_access(self):
        """Protect direct practice/training calls as well as started swarms."""
        task = asyncio.current_task()
        temporary = None
        try:
            if self._checkpoint_owner is None:
                temporary = await self._acquire_checkpoint_owner()
                if temporary is None:
                    self.emit(**self.status())
                    yield False
                    return
            self._owners.add(task)
            yield True
        finally:
            if temporary is not None:
                temporary.close()
            self._owners.discard(task)

    async def _acquire_checkpoint_owner(self):
        future = asyncio.create_task(asyncio.to_thread(TrainingLease.acquire, self.directory))
        owner, interrupted = await finish(future)
        if interrupted:
            if owner is not None:
                owner.close()
            raise asyncio.CancelledError
        return owner

    async def _training_loop(self):
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._force.wait(), timeout=300)
            except asyncio.TimeoutError:
                pass
            self._force.clear()
            while self._live or not admission.ADMISSION.should_run("training"):
                if self._stop.is_set():
                    return
                await asyncio.sleep(1)
            try:
                await self.train_and_validate()
            except Exception as exc:  # noqa: BLE001 - isolate an optional training pass
                self.emit(status="error", reason=str(exc)[:180])

    async def train_and_validate(self):
        if self._training_lock.locked():
            return
        if os.getenv("ORION_GPU_LEASES", "1") != "1":
            self.emit(status="deferred", reason="Training requires exclusive GPU leases")
            return
        async with self._training_lock, self._checkpoint_access() as available:
            if not available:
                return self.status()
            from .weights import active_model
            initial = await asyncio.to_thread(active_model, self.directory)
            checkpoint = self.directory / "training.nnue"
            if await asyncio.to_thread(checkpoint.is_file):
                initial = checkpoint             # continue learning between arena rejections
            async with admission.ADMISSION.slot("training", label="Chess NNUE"):
                async with admission.ADMISSION.lease("gpu", "training", label="Chess NNUE") as ticket:
                    if self._live or not admission.ADMISSION.should_run("training"):
                        return
                    result = await self._run_child(_train_entry,
                        lambda stop, pipe: (str(self.directory), str(initial) if initial else None, stop, pipe),
                        gpu_ticket=ticket, timeout=300)
                # GPU process and its context have exited before this point.
                self.emit(**result)
                if result.get("status") == "candidate" and not self._stop.is_set():
                    from .arena import arena_main
                    candidate = Path(result["path"])
                    try:
                        from .weights import save_training_checkpoint
                        await asyncio.to_thread(save_training_checkpoint, candidate, checkpoint)
                        verdict = await self._run_child(arena_main,
                            lambda stop, pipe: (str(self.brain_path), str(self.directory), str(candidate),
                                                stop, self._pause, pipe), timeout=4 * 3600)
                        self.emit(**verdict)
                    finally:
                        await asyncio.to_thread(candidate.unlink, missing_ok=True)

    async def _book_loop(self, sources):
        from .book import download_pgn
        if len(sources) > 8:
            raise ValueError("At most eight PGN sources")
        while not self._stop.is_set():
            if admission.ADMISSION.should_run("training") and not self._live:
                # Serialise CPU maintenance and training; no extra hidden pool.
                async with self._training_lock:
                    try:
                        paths = [await download_pgn(url, self.directory / "pgn") for url in sources]
                        result = await self._run_child(_book_entry,
                            lambda stop, pipe, paths=paths: (paths, str(self.directory), stop, pipe), timeout=180)
                        self.emit(**result)
                    except Exception as exc:  # noqa: BLE001 - preserve existing book on refresh failure
                        self.emit(status="book error", reason=str(exc)[:180])
            await asyncio.sleep(24 * 3600)

    def shutdown(self):
        self._closed = True
        self._stop.set()
        self.foreground()
        self.search.shutdown()
        self.review.shutdown()
        admission.ADMISSION.unregister_loop(self._loop_name)
        for task in tuple(self._tasks | self._owners):
            task.cancel()
        try:
            return asyncio.get_running_loop().create_task(self.close())
        except RuntimeError:
            # No event loop: daemon workers are reaped by multiprocessing at exit.
            return None

    async def close(self):
        self._closed = True
        self._stop.set()
        self.foreground()
        admission.ADMISSION.unregister_loop(self._loop_name)
        async with self._close_lock:
            # Closing the desktop does not stop an independently scheduled
            # trainer. Only the explicit Stop training control requests that.
            await self.stop_training(include_external=False)
            self._stop.set()
            await self.search.close()
            await self.review.close()
