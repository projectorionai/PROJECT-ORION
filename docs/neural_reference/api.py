"""FastAPI adapter. The existing aiohttp server can call the same core objects."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from pydantic import Field

from .async_utils import complete_before_cancel
from .contracts import Plan, StrictModel


class GoalRequest(StrictModel):
    goal: str = Field(min_length=1, max_length=4000)
    priority: int = Field(default=50, ge=0, le=100)


class JobRequest(StrictModel):
    plan: Plan
    priority: int = Field(default=50, ge=0, le=100)


class ReplanRequest(StrictModel):
    goal: str = Field(min_length=1, max_length=4000)
    priority: int | None = Field(default=None, ge=0, le=100)


class ControlRequest(StrictModel):
    paused: bool | None = None
    priority: int | None = Field(default=None, ge=0, le=100)
    cancel: bool = False


def create_app(engine, memory, scheduler, *, token: str, owner="local-user"):
    if len(token) < 32:
        raise ValueError("Provision a long random local API token")

    async def authenticate(authorization: str = Header(default="")):
        if not secrets.compare_digest(
            authorization.encode("utf-8"), f"Bearer {token}".encode()
        ):
            raise HTTPException(401, "Unauthorised")

    async def watch_pressure():
        high = False
        while True:
            try:
                import psutil

                used = (await asyncio.to_thread(psutil.virtual_memory)).percent
                high = used > (80 if high else 90)  # hysteresis
                app.state.resource_probe_error = ""
            except Exception as exc:  # noqa: BLE001 - resource-provider boundary
                # Lost monitoring must pause dispatch, not kill this task silently.
                high = True
                app.state.resource_probe_error = type(exc).__name__
            await scheduler.set_pressure(high)
            await asyncio.sleep(2)

    @asynccontextmanager
    async def lifespan(app):
        watcher = None
        try:
            await scheduler.start()
            watcher = asyncio.create_task(watch_pressure())
            yield
        finally:

            async def cleanup():
                if watcher is not None:
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
                try:
                    await scheduler.close()
                finally:
                    await asyncio.to_thread(scheduler.journal.close)

            await complete_before_cancel(cleanup())

    app = FastAPI(lifespan=lifespan)
    app.state.resource_probe_error = ""
    router = APIRouter(prefix="/neural", dependencies=[Depends(authenticate)])
    planning = asyncio.Lock()

    def get_job(job_id):
        try:
            return scheduler.owned(job_id, owner)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    async def draft(goal, observations=None):
        if scheduler.closing or scheduler.fatal_error:
            raise HTTPException(503, "Scheduler unavailable")
        # Admission control: do not accumulate arbitrarily many GPU requests.
        if planning.locked():
            raise HTTPException(429, "Planner busy; retry later")
        async with planning:
            try:
                memories = await memory.recall(owner, goal)
                metrics = await asyncio.to_thread(scheduler.journal.metrics)
                return await engine.plan(
                    goal,
                    memories,
                    {"task_state": observations or {}, "tool_metrics": metrics},
                    scheduler.agent.allowed_tools,
                )
            except (ValueError, PermissionError) as exc:
                raise HTTPException(422, type(exc).__name__) from None
            except Exception:  # noqa: BLE001 - sanitise provider/library failures
                raise HTTPException(503, "Planner or memory unavailable") from None

    async def command(operation):
        try:
            return await operation
        except PermissionError:
            raise HTTPException(403, "Action outside authorised scope") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        except Exception:  # noqa: BLE001 - journal/provider exception boundary
            raise HTTPException(503, "Scheduler unavailable") from None

    @router.post("/plans")
    async def propose(request: GoalRequest):
        return await draft(request.goal)

    @router.post("/goals", status_code=202)
    async def submit_goal(request: GoalRequest):
        plan = await draft(request.goal)
        job_id = await command(scheduler.submit(owner, plan, request.priority))
        return {"job_id": job_id, "plan": plan}

    @router.post("/jobs", status_code=202)
    async def submit_plan(request: JobRequest):
        return {
            "job_id": await command(
                scheduler.submit(owner, request.plan, request.priority)
            )
        }

    @router.get("/jobs/{job_id}")
    async def status(job_id: str):
        return get_job(job_id).snapshot()

    @router.patch("/jobs/{job_id}")
    async def control(job_id: str, request: ControlRequest):
        get_job(job_id)
        await command(scheduler.control(job_id, owner, **request.model_dump()))
        return get_job(job_id).snapshot()

    @router.post("/jobs/{job_id}/replan")
    async def replan(job_id: str, request: ReplanRequest):
        job = get_job(job_id)
        if not job.paused or any(
            v in {"running", "uncertain"} for v in job.states.values()
        ):
            raise HTTPException(409, "Pause and wait for a quiescent state")
        before = job.snapshot()
        plan = await draft(
            request.goal + " Preserve every succeeded step exactly, with the same ID.",
            {"previous_job": before},
        )
        await command(
            scheduler.revise(
                job_id,
                owner,
                plan,
                expected_revision=before["revision"],
                priority=request.priority,
            )
        )
        return get_job(job_id).snapshot()  # remains paused until explicitly resumed

    @router.get("/telemetry")
    async def telemetry():
        return {
            "tools": await command(asyncio.to_thread(scheduler.journal.metrics)),
            "resource_pressure": scheduler.pressure,
            "resource_probe_error": app.state.resource_probe_error,
            "scheduler_error": scheduler.fatal_error,
        }

    app.include_router(router)
    return app
