"""
Benchmark ORION's streaming local voice loop (local brain, stage 2) - DRY.

    python -m tools.eval_local_voice                     # 12 utterances x 2 runs, llama3.1:latest
    python -m tools.eval_local_voice --runs 3 --model qwen2.5:7b
    python -m tools.eval_local_voice --wav-dir C:\\recordings  # your own 16 kHz mono WAVs
    python -m tools.eval_local_voice --no-compare        # streaming only (half the time)

One case goes through what the offline voice loop does, minus the microphone
and the speakers:

    utterance WAV -> OfflineTranscriber (local Whisper/Vosk) -> LocalAgent (real
    shortlist, real Ollama stream, DRY-RUN dispatcher: nothing is executed) ->
    sentence streaming -> Kokoro synthesis of every chunk (PCM made, never played)

Measured per case, from the END of the owner's speech:

    endpoint_ms     the configured silence that ends an utterance (added, not slept)
    stt_ms          transcription
    first_token_ms  the model's first fragment (from the agent's start)
    first_chunk_ms  the first speakable chunk handed to the voice (agent start)
    ttfa_ms         end of speech -> first audio chunk synthesised
                    = endpoint + stt + (agent start -> first PCM)
    total_ms        end of speech -> every chunk synthesised

With --compare (the default) each case is also answered the stage-1 way -
generate the whole reply, then synthesise it - so the report has before and
after on the same machine, model and utterances.

The synthetic utterances are made with Kokoro itself, in a different voice
from ORION's, into a temporary folder that is deleted afterwards (--keep keeps
it). The tool never pulls a model and never starts Ollama: Ollama must already
be running with the model present, and it says so plainly when it is not.
Writes JSON (every row) and a Markdown summary to reports/local_voice_eval/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_local_agent import (  # noqa: E402
    OLLAMA, DryRunDispatcher, local_router, ollama_models, percentile)

REPORT_FORMAT = "orion-local-voice-eval-v1"
SAMPLE_RATE = 16000
#: Owner-side utterances: (id, kind, words). "chat" needs no tool; "tool" does.
UTTERANCES: tuple[tuple[str, str, str], ...] = (
    ("hello", "chat", "Hello Orion, how are you today?"),
    ("who", "chat", "Who are you?"),
    ("thanks", "chat", "Thanks, that's all for now."),
    ("focus-tip", "chat", "Give me a quick tip for staying focused."),
    ("moon", "chat", "Tell me something interesting about the moon."),
    ("joke", "chat", "Tell me a short joke."),
    ("time", "tool", "What time is it?"),
    ("notepad", "tool", "Open Notepad."),
    ("memory", "tool", "How much memory is my computer using?"),
    ("percent", "tool", "What's fifteen percent of two hundred and forty?"),
    ("timer", "tool", "Set a timer for five minutes."),
    ("music", "tool", "Can you pause the music?"),
)


# ── audio files ─────────────────────────────────────────────────────────────

def resample_16k(samples: Any, rate: int) -> Any:
    import numpy as np
    audio = np.asarray(samples, dtype=np.float32).ravel()
    if not rate or rate == SAMPLE_RATE or not audio.size:
        return audio
    positions = np.linspace(0, audio.size - 1, int(round(audio.size * SAMPLE_RATE / rate)))
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


def write_wav(path: Path, samples: Any, *, lead_s: float = 0.3) -> float:
    """16 kHz mono 16-bit WAV with a little silence before the words and none
    after: the end of the file is the end of speech. Returns its seconds."""
    import numpy as np
    audio = np.concatenate([np.zeros(int(lead_s * SAMPLE_RATE), dtype=np.float32),
                            np.asarray(samples, dtype=np.float32).ravel()])
    peak = float(np.max(np.abs(audio))) or 1.0
    pcm = (np.clip(audio * min(1.0, 0.8 / peak), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return len(pcm) / SAMPLE_RATE


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path.name}: needs mono 16-bit PCM")
        return wav.readframes(wav.getnframes()), wav.getframerate()


def synthesise_utterances(make: Callable[[str], tuple[Any, int]], folder: Path,
                          utterances: Sequence[tuple[str, str, str]] = UTTERANCES
                          ) -> list[dict[str, Any]]:
    """Each utterance spoken by *make* (text -> (samples, rate)) into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id, kind, words in utterances:
        samples, rate = make(words)
        path = folder / f"{case_id}.wav"
        seconds = write_wav(path, resample_16k(samples, rate))
        rows.append({"id": case_id, "kind": kind, "words": words, "path": str(path),
                     "seconds": round(seconds, 2)})
    return rows


def kokoro_dirs(choice: str = "") -> list[Path] | None:
    """Where the Kokoro model is read from: --kokoro-dir; else ORION's own
    search order (None); else, from a source checkout, the built app's bundled
    copy (dist/ORION/_internal/assets/models). Read only."""
    from orion_core import voice_kokoro
    if choice:
        return [Path(choice)]
    if voice_kokoro.model_dir() is not None:
        return None
    bundled = ROOT / "dist" / "ORION" / "_internal" / "assets" / "models" / voice_kokoro.MODEL.name
    return [bundled] if bundled.is_dir() else None


def kokoro_maker(speaker: str, dirs: list[Path] | None = None
                 ) -> tuple[Any, Callable[[str], tuple[Any, int]]]:
    """(the loaded KokoroVoice, a text -> (samples, rate) function in *speaker*'s voice)."""
    from orion_core import voice_kokoro
    voice_kokoro.enable()
    voice = voice_kokoro.KokoroVoice(dirs=dirs)
    if not voice.available():
        raise RuntimeError(voice.describe())
    engine = voice.load()
    lang = voice_kokoro.language_for(speaker)

    def make(text: str) -> tuple[Any, int]:
        samples, rate = engine.create(text, voice=speaker, speed=1.0, lang=lang)
        return samples, int(rate)

    return voice, make


# ── the speaking side: a TTS worker like ORION's (its own thread) ───────────

class SynthesisWorker(threading.Thread):
    """Synthesises each submitted chunk in order, on its own thread (as
    SpeechSynthesiser does), noting when the first PCM appears. Plays nothing."""

    def __init__(self, voice: Any, clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(name="eval-voice-synthesis", daemon=True)
        self.voice = voice
        self.clock = clock
        self.queue: Queue[str | None] = Queue()
        self.chunks: list[str] = []
        self.first_submit_at: float | None = None
        self.first_pcm_at: float | None = None
        self.done_at: float | None = None
        self.audio_bytes = 0
        self.errors: list[str] = []

    def submit(self, text: str) -> None:
        if self.first_submit_at is None:
            self.first_submit_at = self.clock()
        self.chunks.append(text)
        self.queue.put(text)

    def finish(self) -> None:
        self.queue.put(None)

    def _on_pcm(self, chunk: bytes) -> None:
        if self.first_pcm_at is None and chunk:
            self.first_pcm_at = self.clock()
        self.audio_bytes += len(chunk or b"")

    def run(self) -> None:
        while True:
            text = self.queue.get()
            if text is None:
                break
            try:
                if self.voice.stream_pcm(text, on_chunk=self._on_pcm) is None:
                    self.errors.append(f"no audio for {text[:40]!r}")
            except Exception as exc:          # a measurement, not a crash
                self.errors.append(f"{type(exc).__name__}: {exc}"[:200])
        self.done_at = self.clock()


# ── one case ────────────────────────────────────────────────────────────────

@dataclass
class VoiceBench:
    """The loop under measurement, with every part injectable (the suite runs
    it on fakes): *transcribe* (pcm, rate) -> text; *agent* () -> a fresh
    LocalAgent-like object; *tts* () -> a SynthesisWorker-like object."""

    transcribe: Callable[[bytes, int], str]
    agent: Callable[[], Any]
    tts: Callable[[], Any]
    endpoint_ms: float
    clock: Callable[[], float] = time.perf_counter
    rows: list[dict[str, Any]] = field(default_factory=list)

    async def case(self, case_id: str, kind: str, pcm: bytes, rate: int, *,
                   streamed: bool = True) -> dict[str, Any]:
        stt_started = self.clock()
        heard = await asyncio.to_thread(self.transcribe, pcm, rate)
        stt_ms = (self.clock() - stt_started) * 1000.0
        worker = self.tts()
        worker.start()
        agent = self.agent()
        agent_started = self.clock()
        try:
            if streamed:
                reply = await agent.respond(heard, origin="voice", speak=worker.submit)
            else:
                reply = await agent.respond(heard, origin="voice")
                if getattr(reply, "text", ""):
                    worker.submit(reply.text)            # stage 1: the whole reply at once
        finally:
            generated = self.clock()
            worker.finish()
            await asyncio.to_thread(worker.join, 120.0)
        timings = dict(getattr(reply, "timings", {}) or {})
        first_pcm = getattr(worker, "first_pcm_at", None)
        done = getattr(worker, "done_at", None) or self.clock()
        base = self.endpoint_ms + stt_ms
        row = {
            "id": case_id, "kind": kind, "mode": "streamed" if streamed else "whole",
            "heard": heard, "reply": str(getattr(reply, "text", ""))[:300],
            "spoken": bool(getattr(reply, "spoken", False)),
            "reason": str(getattr(reply, "reason", "")),
            "tools": [c.get("name") for c in (getattr(reply, "tool_calls", None) or [])],
            "chunks": len(getattr(worker, "chunks", []) or []),
            "endpoint_ms": round(self.endpoint_ms, 1),
            "stt_ms": round(stt_ms, 1),
            "first_token_ms": _round(timings.get("first_token_ms")),
            "first_chunk_ms": _round(timings.get("first_chunk_ms")),
            "generation_ms": round((generated - agent_started) * 1000.0, 1),
            "ttfa_ms": (None if first_pcm is None
                        else round(base + (first_pcm - agent_started) * 1000.0, 1)),
            "total_ms": round(base + (done - agent_started) * 1000.0, 1),
            "audio_s": round(getattr(worker, "audio_bytes", 0) / 2 / 24000.0, 2),
            "tts_errors": list(getattr(worker, "errors", []) or []),
        }
        self.rows.append(row)
        return row


def _round(value: Any) -> float | None:
    return None if not isinstance(value, (int, float)) else round(float(value), 1)


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """p50/p95 of every measure, per mode, overall and for each kind."""
    out: dict[str, Any] = {}
    for mode in ("streamed", "whole"):
        mine = [r for r in rows if r.get("mode") == mode]
        if not mine:
            continue
        block: dict[str, Any] = {"cases": len(mine)}
        for group, members in (("all", mine), *[(kind, [r for r in mine if r.get("kind") == kind])
                                                for kind in ("chat", "tool")]):
            if not members:
                continue
            block[group] = {key: {"p50": percentile([r.get(key) for r in members], 0.5),
                                  "p95": percentile([r.get(key) for r in members], 0.95)}
                            for key in ("stt_ms", "first_token_ms", "first_chunk_ms",
                                        "ttfa_ms", "total_ms")}
            block[group]["cases"] = len(members)
        out[mode] = block
    return out


def markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [f"# Local voice loop - {report['model']} ({report['created']})", "",
             f"STT {report.get('stt_engine')} ({report.get('stt_model')}), voice Kokoro "
             f"{report.get('tts_voice')}, endpoint {report.get('endpoint_ms')} ms (configured, "
             f"added), {report.get('cases')} utterances x {report.get('runs')} runs. "
             "Times are from the end of the owner's speech.", "",
             "| Mode | Group | n | STT p50 | First token p50 | First chunk p50 | "
             "**First audio p50 / p95** | Total p50 |", "|---|---|---|---|---|---|---|---|"]
    for mode in ("streamed", "whole"):
        block = summary.get(mode) or {}
        for group in ("all", "chat", "tool"):
            stats = block.get(group)
            if not stats:
                continue

            def p(key: str, q: str = "p50") -> Any:
                return (stats.get(key) or {}).get(q)

            lines.append(f"| {mode} | {group} | {stats.get('cases')} | {p('stt_ms')} | "
                         f"{p('first_token_ms')} | {p('first_chunk_ms')} | "
                         f"**{p('ttfa_ms')} / {p('ttfa_ms', 'p95')}** | {p('total_ms')} |")
    warm = report.get("warm") or {}
    if warm:
        lines += ["", "Warm-up (before the cases): " + ", ".join(
            f"{name} {value}" for name, value in warm.items())]
    return "\n".join(lines) + "\n"


# ── the run ─────────────────────────────────────────────────────────────────

def _preflight(args: argparse.Namespace) -> list[str]:
    """Everything missing, in plain words ([] = ready). Loads nothing."""
    problems = []
    try:
        present = ollama_models(args.ollama)
        if args.model not in present:
            problems.append(f"model '{args.model}' is not in Ollama (present: "
                            f"{', '.join(sorted(present)) or 'none'}); this tool never pulls "
                            "one - run `ollama pull` yourself")
    except Exception as exc:
        problems.append(f"Ollama is not answering at {args.ollama} ({exc}); start it yourself "
                        "(`ollama serve`) - this tool never starts or pulls anything")
    from orion_core import voice_kokoro
    voice_kokoro.enable()
    probe = voice_kokoro.KokoroVoice(dirs=kokoro_dirs(getattr(args, "kokoro_dir", "")))
    if not probe.available():
        problems.append(f"the Kokoro voice is not usable here: {probe.describe()}")
    from orion_core.speech_offline import OfflineTranscriber
    if not OfflineTranscriber(defer=True).cached_available:
        problems.append("no cached offline Whisper model (ORION_WHISPER_MODEL, default base.en)")
    return problems


async def run(args: argparse.Namespace) -> int:
    problems = _preflight(args)
    if problems:
        for line in problems:
            print(f"NOT READY: {line}", file=sys.stderr)
        return 2
    from orion_core.local_agent import LocalAgent
    from orion_core.speech_offline import OfflineTranscriber
    from orion_core.voice_stream import local_endpoint_s

    report: dict[str, Any] = {"format": REPORT_FORMAT, "model": args.model,
                              "created": datetime.now().isoformat(timespec="seconds"),
                              "runs": args.runs, "compare": not args.no_compare}
    folder = Path(args.wav_dir) if args.wav_dir else Path(tempfile.mkdtemp(prefix="orion_voice_eval_"))
    made_here = not args.wav_dir
    router = local_router(args.ollama)
    try:
        voice, make = kokoro_maker(args.speaker, kokoro_dirs(args.kokoro_dir))
        if made_here:
            print(f"making {len(UTTERANCES)} utterances in {args.speaker}'s voice -> {folder}",
                  flush=True)
            cases = synthesise_utterances(make, folder)
        else:
            cases = [{"id": p.stem, "kind": "recorded", "words": "", "path": str(p)}
                     for p in sorted(folder.glob("*.wav"))]
            if not cases:
                print(f"No .wav files in {folder}.", file=sys.stderr)
                return 2
        transcriber = OfflineTranscriber(defer=True)
        dry = DryRunDispatcher()

        def new_agent() -> LocalAgent:
            dry.reset()
            return LocalAgent(router, dry, dispatch=dry.dispatch,
                              declarations=lambda: dry.TOOL_DECLARATIONS, model=args.model,
                              keep_alive="10m", deadline_s=args.deadline)

        print("warming: offline hearing, the local model, the voice…", flush=True)
        warm: dict[str, Any] = {}
        hearing = await asyncio.to_thread(transcriber.warm)
        warm["stt"] = f"{hearing.get('engine')} load {hearing.get('load_ms')} ms"
        model = await new_agent().warm()
        warm["model"] = (f"{model.get('ms')} ms (load {model.get('load_ms')})" if model.get("ok")
                         else f"failed: {model.get('why')}")
        if not model.get("ok"):
            print(f"NOT READY: the local model did not answer - {model.get('why')}", file=sys.stderr)
            return 2
        started = time.perf_counter()
        await asyncio.to_thread(voice.stream_pcm, "Ready.", on_chunk=lambda _c: None)
        warm["tts"] = f"{(time.perf_counter() - started) * 1000.0:.0f} ms"
        report.update(warm=warm, stt_engine=transcriber.engine, stt_model=transcriber.model_size,
                      tts_voice=voice.voice, cases=len(cases))
        endpoint_ms = (local_endpoint_s() * 1000.0) if args.endpoint_ms is None else args.endpoint_ms
        report["endpoint_ms"] = round(endpoint_ms, 1)

        def transcribe(pcm: bytes, rate: int) -> str:
            return transcriber.transcribe_pcm(pcm, sample_rate=rate, identity_checked=True)

        bench = VoiceBench(transcribe=transcribe, agent=new_agent,
                           tts=lambda: SynthesisWorker(voice), endpoint_ms=endpoint_ms)
        modes = [True] if args.no_compare else [True, False]
        for run_no in range(1, args.runs + 1):
            # Each mode is a block over every utterance, and the block that
            # goes first alternates by run. Running both modes back to back on
            # one utterance (25 Oct) sent the second the identical prompt, which
            # Ollama answered from its prompt cache: the whole replies' model
            # time was 704 ms against 1,750 ms streamed, and streaming looked
            # slower. With blocks (26 Oct) the first tokens matched.
            order = modes if run_no % 2 else modes[::-1]
            for streamed in order:
                for case in cases:
                    pcm, rate = read_wav(Path(case["path"]))
                    row = await bench.case(case["id"], case["kind"], pcm, rate, streamed=streamed)
                    row["run"] = run_no
                    print(f"  run {run_no} {row['mode']:8} {case['id']:10} heard {row['heard'][:40]!r:44} "
                          f"first audio {row['ttfa_ms']} ms, total {row['total_ms']} ms", flush=True)
        report["rows"] = bench.rows
        report["summary"] = summarise(bench.rows)
    finally:
        try:
            await router.unload_local_models("voice evaluation finished")
        except Exception:
            pass
        await router.aclose()
        if made_here and not args.keep:
            shutil.rmtree(folder, ignore_errors=True)
    out = Path(args.out) if args.out else (
        ROOT / "reports" / "local_voice_eval"
        / f"local_voice_{args.model.replace(':', '_').replace('/', '_')}_"
          f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    table = markdown(report)
    out.with_suffix(".md").write_text(table, encoding="utf-8")
    print(table)
    print(f"Written: {out} and {out.with_suffix('.md')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.eval_local_voice",
                                     description="Benchmark the streaming local voice loop (dry).")
    parser.add_argument("--model", default="llama3.1:latest",
                        help="an Ollama tag that is ALREADY pulled (never downloaded)")
    parser.add_argument("--runs", type=int, default=2, help="passes over the utterances")
    parser.add_argument("--no-compare", action="store_true",
                        help="skip the stage-1 (whole reply, then speech) comparison")
    parser.add_argument("--speaker", default="bm_lewis",
                        help="the Kokoro voice that plays the owner in synthetic utterances")
    parser.add_argument("--wav-dir", default="", help="use these 16 kHz mono WAVs instead")
    parser.add_argument("--kokoro-dir", default="",
                        help="the kokoro-v1.0 model folder (default: ORION's search, then "
                             "dist/ORION/_internal/assets/models)")
    parser.add_argument("--keep", action="store_true", help="keep the synthetic WAV folder")
    parser.add_argument("--endpoint-ms", type=float, default=None,
                        help="endpoint silence to add (default: the local brain's, 650 ms)")
    parser.add_argument("--deadline", type=float, default=30.0, help="per-turn budget (s)")
    parser.add_argument("--ollama", default=OLLAMA, help="Ollama root URL")
    parser.add_argument("--out", default="", help="JSON report path (Markdown written beside it)")
    args = parser.parse_args(argv)
    args.runs = max(1, int(args.runs))
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
