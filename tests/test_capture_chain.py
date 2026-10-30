"""Synthetic PCM and fake native processors: no microphone, sound or downloads.

Dedicated coverage is necessary because the existing audio tests exercise
playback/state machinery, whereas capture DSP has stateful 160-sample frames,
sample clocks and reference alignment. These tests cannot certify native AEC
quality, owner recognition accuracy or real-device interruption latency.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from orion_core import audio
from orion_core import capture_chain as dsp


def _pcm(x):
    return np.asarray(x, dtype="<i2").tobytes()


def _passthrough():
    return dsp.CaptureChain(aec=False, ns=False, agc=False, hpf=False)


def test_frame_delay_and_sample_count_are_continuous_across_512_sample_chunks():
    source = np.arange(4096, dtype="<i2")
    chain = _passthrough()
    pieces = [chain.process(_pcm(source[i:i + 512]), (i + 512) / dsp.SAMPLE_RATE)
              for i in range(0, source.size, 512)]
    result = np.frombuffer(b"".join(pieces), dtype="<i2")
    assert result.size == source.size
    assert np.array_equal(result[:dsp.FRAME], np.zeros(dsp.FRAME))
    assert np.array_equal(result[dsp.FRAME:], source[:-dsp.FRAME])


def test_a_capture_gap_does_not_replay_the_old_partial_frame():
    chain = _passthrough()
    chain.process(_pcm(np.full(512, 11000)), 1.032)
    result = np.frombuffer(chain.process(_pcm(np.full(512, -1200)), 9.032), dtype="<i2")
    assert np.all(result[:dsp.FRAME] == 0)
    assert np.all(result[dsp.FRAME:] == -1200)
    assert 11000 not in result


def test_a_fault_passes_raw_pcm_and_retires_buffered_speech(monkeypatch):
    chain = _passthrough()
    chain._outq = np.full(dsp.FRAME, 12345, dtype=np.float32)
    chain._pending = np.full(17, 23456, dtype=np.float32)

    def broken(_chunk, _at):
        chain._clock.count += 512
        raise RuntimeError("synthetic processor failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(chain, "_process", broken)
        chunk = _pcm(np.full(512, 600))
        assert chain.process(chunk, 1.0) == chunk
    assert chain._clock.count == 0
    assert chain._pending.size == 0
    result = np.frombuffer(chain.process(_pcm(np.full(512, 700)), 1.032), dtype="<i2")
    assert np.all(result[:dsp.FRAME] == 0)
    assert not np.isin(result, [12345, 23456]).any()


def test_three_consecutive_faults_disable_processing_without_losing_hearing(monkeypatch):
    chain = _passthrough()
    monkeypatch.setattr(chain, "_process", lambda *_: (_ for _ in ()).throw(ValueError("bad DSP")))
    chunk = _pcm([500] * 512)
    for _ in range(4):
        assert chain.process(chunk) == chunk
    assert not chain.active
    assert not chain.echo_free()


def test_missing_native_module_reports_no_cancellation_or_suppression(monkeypatch):
    monkeypatch.setitem(sys.modules, "aec_audio_processing", None)
    logs = []
    chain = dsp.CaptureChain(log=logs.append, agc=False, hpf=False)
    assert len(chain.process(_pcm([10] * 512), 1.0)) == 1024
    assert chain.stats()["cancelling"] is False
    assert chain.stats()["noise_suppression"] is False
    assert any("unavailable" in message for message in logs)
    chain.converged = True
    chain.near_end_at = 1.0
    assert not chain.echo_free(1.0)


@pytest.mark.parametrize("correlated", [True, False])
def test_suppression_or_unrelated_input_cannot_falsely_attest_aec(monkeypatch, correlated):
    class Processor:
        def __init__(self, enable_aec=False, enable_ns=False, **_):
            self.ns = enable_ns
            self.aec = enable_aec

        def set_stream_format(self, *_):
            pass

        def set_reverse_stream_format(self, *_):
            pass

        def process_reverse_stream(self, _):
            pass

        def process_stream(self, data):
            if self.ns:
                return bytes(len(data))
            # An identity AEC plus strong NS cannot claim cancellation. Nor
            # can an attenuator with no matching speaker-reference content.
            if correlated:
                return data
            return _pcm(np.frombuffer(data, dtype="<i2") // 100)

    monkeypatch.setitem(sys.modules, "aec_audio_processing", SimpleNamespace(AudioProcessor=Processor))
    rng = np.random.default_rng(12)
    reference = rng.integers(-8000, 8000, 8192, dtype=np.int16)
    microphone = reference if correlated else rng.integers(-8000, 8000, reference.size, dtype=np.int16)
    playback = dsp.PlaybackReference()
    playback.note(_pcm(reference), at=0.0, sample_rate=dsp.SAMPLE_RATE)
    chain = dsp.CaptureChain(reference=playback, agc=False, hpf=False)
    chain.delay_s = 0.0
    for start in range(0, reference.size, 512):
        chain.process(_pcm(microphone[start:start + 512]), (start + 512) / dsp.SAMPLE_RATE)
    assert not chain.converged
    assert not chain.echo_free(reference.size / dsp.SAMPLE_RATE)


def test_an_unmeasured_delay_is_not_cancellation_evidence(monkeypatch):
    chain = _passthrough()
    chain.aec_wanted = True
    chain._apm = object()
    source = np.random.default_rng(1).integers(-5000, 5000, 16000).astype(np.float32)
    chain._ref_tl.write(0, source)
    chain._update_echo(source, source * 0.01, 0, 100, 1.0)
    assert chain._erle == dsp.deque()
    assert not chain.converged


def test_delay_estimator_requires_agreement_on_the_actual_synthetic_lag():
    lag = 2400
    source = np.random.default_rng(81).normal(0, 3500, 64000).astype(np.float32)
    reference, mic = dsp._Timeline(8.0), dsp._Timeline(8.0)
    reference.write(0, source)
    mic.write(lag, source)
    estimator = dsp._DelayEstimator()
    assert estimator.run(mic, reference, 48000) is None
    assert estimator.run(mic, reference, 56000) == pytest.approx(lag, abs=48)


def test_invalid_capture_timestamp_fails_open_to_raw_hearing():
    chain = _passthrough()
    chunk = _pcm([500] * 512)
    assert chain.process(chunk, float("nan")) == chunk
    assert chain._clock.count == 0


def test_callback_preserves_adc_timing_and_spacing_before_queue_delay(monkeypatch):
    captured = []
    engine = audio.MicrophoneEngine.__new__(audio.MicrophoneEngine)
    engine.enabled = True
    engine._gatekeeper = SimpleNamespace(enqueue_capture=lambda pcm, at: captured.append((pcm, at)))
    monkeypatch.setattr(audio.time, "monotonic", lambda: 100.0)
    source = _pcm(np.arange(2048))
    engine._callback(source, 2048, SimpleNamespace(inputBufferAdcTime=20.0, currentTime=20.05), None)
    assert b"".join(pcm for pcm, _ in captured) == source
    assert [at for _, at in captured] == pytest.approx([99.982, 100.014, 100.046, 100.078])


def test_legacy_callback_consumers_still_receive_plain_pcm(monkeypatch):
    captured = []
    engine = audio.MicrophoneEngine.__new__(audio.MicrophoneEngine)
    engine.enabled = True
    engine._gatekeeper = SimpleNamespace(enqueue=captured.append)
    source = _pcm(np.arange(2048))
    engine._callback(source, 2048, None, None)
    assert b"".join(captured) == source


@pytest.mark.parametrize("fails", [False, True])
def test_playback_reference_only_contains_slices_written_to_the_device(monkeypatch, fails):
    reference = dsp.PlaybackReference()
    monkeypatch.setattr(dsp, "speaker_reference", lambda: reference)
    monkeypatch.setattr(dsp, "enabled", lambda: True)

    def write(_):
        if fails:
            raise RuntimeError("synthetic failed speaker write")

    playback = audio.AudioPlaybackThread(SimpleNamespace())
    playback._stream = SimpleNamespace(write=write)
    source = _pcm([200] * 256)
    if fails:
        with pytest.raises(RuntimeError):
            playback._write(source)
        assert reference.since(0) == []
    else:
        playback._write(source)
        assert b"".join(bytes(item[3]) for item in reference.since(0)) == source


def test_gate_recognises_cleaned_pcm_but_identity_and_watchdog_see_raw(monkeypatch):
    from orion_core import voice_enrolment

    raw, cleaned = _pcm([2000] * 512), _pcm([12] * 512)
    received, recognised, genders, logs = [], [], [], []
    bus = SimpleNamespace(log=SimpleNamespace(emit=logs.append), amplitude=SimpleNamespace(emit=lambda *_: None))
    gate = audio.AudioGateThread(
        SimpleNamespace(call_soon_threadsafe=lambda fn, *args: fn(*args)),
        SimpleNamespace(full=lambda: False, put_nowait=received.append),
        bus, lambda: True, SimpleNamespace(confidence=lambda pcm: 0.99 if pcm == cleaned else 0.0, threshold=0.5),
        recogniser=SimpleNamespace(feed=lambda pcm: recognised.append(pcm) or ""),
        on_transcript=lambda _: None, speaking_check=lambda: False,
        speaker_tracker=SimpleNamespace(feed=lambda pcm, _: genders.append(pcm)))
    monkeypatch.setattr(gate, "_clean_capture", lambda pcm, at: cleaned)
    monkeypatch.setattr(gate, "_speaker_judged", lambda: False)
    monkeypatch.setattr(audio.time, "monotonic", lambda: 500.0)
    monkeypatch.setattr(voice_enrolment, "holding_model", lambda: False)
    gate._last_signal_at = 0.0
    gate.enqueue_capture(raw, 500.0)
    gate.raw_q.put(None)
    gate.run()
    assert recognised == [cleaned]
    assert received[0]["data"] == cleaned
    assert genders == [raw]
    assert gate._presence.utterance.audio() == raw
    assert gate._last_signal_at == 500.0
    assert not any("recovered" in message for message in logs)


def test_opt_out_restores_untouched_capture(monkeypatch):
    monkeypatch.setenv("ORION_CAPTURE_DSP", "0")
    gate = audio.AudioGateThread.__new__(audio.AudioGateThread)
    raw = _pcm([1800] * 512)
    assert gate._clean_capture(raw, 1.0) == raw
    assert gate._capture_chain is None


def test_future_near_end_timestamp_cannot_open_echo_guard():
    chain = _passthrough()
    chain._apm = object()
    chain.aec_wanted = True
    chain.converged = True
    chain.near_end_at = 20.0
    assert not chain.echo_free(19.0)


@pytest.mark.parametrize("first_jitter", [0, 128])
def test_mic_clock_keeps_counted_spacing_when_callback_jitter_moves_window(first_jitter):
    clock = dsp._MicClock()
    n = 512
    jitters = [first_jitter] + [32, 144, 48, 112] * 20
    starts = [clock.place(n, (n * (i + 1) + jitter) / dsp.SAMPLE_RATE)
              for i, jitter in enumerate(jitters)]
    assert starts == [n * i + first_jitter for i in range(len(jitters))]


def test_mic_clock_ignores_short_lateness_bursts():
    clock = dsp._MicClock()
    n = 512
    starts = [clock.place(n, n / dsp.SAMPLE_RATE)]
    jitters = [640] * 12 + [0] * 4
    starts.extend(clock.place(n, (n * (i + 2) + jitter) / dsp.SAMPLE_RATE)
                  for i, jitter in enumerate(jitters))
    assert starts == [n * i for i in range(len(starts))]


@pytest.mark.parametrize("jitter", [-320, 320])
def test_mic_clock_keeps_alignment_at_the_twenty_ms_tolerance(jitter):
    clock = dsp._MicClock()
    n = 160
    clock.place(n, (16000 + n) / dsp.SAMPLE_RATE)
    for i in range(80):
        start = clock.place(n, (16000 + n * (i + 2) + jitter) / dsp.SAMPLE_RATE)
        assert start == 16000 + n * (i + 1)


@pytest.mark.parametrize("n", [160, 512])
def test_mic_clock_reanchors_after_persistent_lateness(n):
    clock = dsp._MicClock()
    clock.place(n, n / dsp.SAMPLE_RATE)
    chunks = (8000 + n - 1) // n
    starts = [clock.place(n, (n * (i + 2) + 640) / dsp.SAMPLE_RATE)
              for i in range(chunks)]
    assert starts[:-1] == [n * (i + 1) for i in range(chunks - 1)]
    assert starts[-1] == n * chunks + 640
    assert clock.place(n, (n * (chunks + 2) + 640) / dsp.SAMPLE_RATE) == starts[-1] + n


@pytest.mark.parametrize("jump", [4800, -480])
def test_mic_clock_reanchors_immediately_on_a_true_discontinuity(jump):
    clock = dsp._MicClock()
    n = 512
    clock.place(n, (16000 + n) / dsp.SAMPLE_RATE)
    start = clock.place(n, (16000 + 2 * n + jump) / dsp.SAMPLE_RATE)
    assert start == 16000 + n + jump
    assert clock.place(n, (16000 + 3 * n + jump) / dsp.SAMPLE_RATE) == start + n


@pytest.mark.parametrize("first_jitter", [0, 128])
def test_callback_jitter_cannot_shift_pcm_against_its_exact_echo_reference(first_jitter):
    # An ideal subtractor proves alignment only, not native AEC quality. Both
    # the PCM and speaker timestamps are synthetic and no device is opened.
    class ExactReferenceSubtractor:
        def process_reverse_stream(self, data):
            self.reference = np.frombuffer(data, dtype="<i2").astype(np.int32)

        def process_stream(self, data):
            mic = np.frombuffer(data, dtype="<i2").astype(np.int32)
            return _pcm(np.clip(mic - self.reference, -32768, 32767))

    n = 512
    jitters = [first_jitter] + [32, 144, 48, 112] * 20
    source = np.random.default_rng(20261012).integers(-3000, 3000, n * len(jitters), dtype=np.int16)
    reference = dsp.PlaybackReference()
    reference.note(_pcm(source), at=0.0, sample_rate=dsp.SAMPLE_RATE)
    chain = dsp.CaptureChain(reference=reference, aec=True, ns=False, agc=False, hpf=False)
    chain._ready = True
    chain._apm = ExactReferenceSubtractor()
    chain.d_pre = first_jitter
    chain._estimator = SimpleNamespace(due=lambda _: False, w=16000, max_lag=12800)
    pieces = [chain.process(_pcm(source[i * n:(i + 1) * n]),
                            (n * (i + 1) + jitter) / dsp.SAMPLE_RATE)
              for i, jitter in enumerate(jitters)]
    output = np.frombuffer(b"".join(pieces), dtype="<i2")
    assert output.size == source.size
    assert np.count_nonzero(output) == 0


def test_capture_clock_reanchor_retires_partial_audio_and_aec_attestation():
    chain = _passthrough()
    n = 512
    chain.process(_pcm([11000] * n), 1.0 + n / dsp.SAMPLE_RATE)
    chunks = (8000 + n - 1) // n
    for i in range(chunks - 1):
        chain.process(_pcm([-1200] * n), 1.0 + n * (i + 2) / dsp.SAMPLE_RATE + 0.04)
    chain.converged = True
    chain._erle.extend([25.0] * 40)
    chain.d_pre = 640
    chain._estimator.lag = 640
    result = np.frombuffer(chain.process(_pcm([-1200] * n),
                                        1.0 + n * (chunks + 1) / dsp.SAMPLE_RATE + 0.04),
                           dtype="<i2")
    assert np.all(result[:dsp.FRAME] == 0)
    assert np.all(result[dsp.FRAME:] == -1200)
    assert not chain.converged
    assert not chain._erle
    assert chain._estimator.lag is None
    assert chain.d_pre == 0
