"""ORION's voice choice (voice_settings.py), Mark XXXVII.

The owner first wanted a cute voice to match the Screen Face; Mark XXXIX
retired the cute style ("professional and not cute"), and he picks voices
from samples. So the voice became a setting - kokoro_voice, kokoro_speed,
cloud_voice - read by the offline voice (voice_kokoro) and by the Live /
Charon / phone speech paths. The DEFAULTS DID NOT CHANGE: Charon and bm_george.

Fakes only: no model is loaded, nothing is played, no network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import voice_kokoro as vk  # noqa: E402
from orion_core import voice_settings as vs  # noqa: E402


# ── what exists, and the unchanged defaults ──────────────────────────────────

def test_the_defaults_are_unchanged_charon_and_george():
    assert vs.DEFAULTS == {"kokoro_voice": "bm_george", "kokoro_speed": 1.0, "cloud_voice": "Charon",
                           "reply_pause": "natural", "expressive": True}
    assert vs.snapshot() == vs.DEFAULTS                      # nothing stored, nothing set
    from orion_core.constants import VOICE_PROFILE

    assert vs.CLOUD_DEFAULT_VOICE == VOICE_PROFILE.gemini_voice_name == "Charon"
    assert vs.KOKORO_DEFAULT_VOICE == vk.DEFAULT_VOICE == "bm_george"
    assert vk.current_voice() == "bm_george" and vk.base_speed() == 1.0
    assert vs.current_style() == "classic"


def test_the_stores_are_redirected_for_every_test(tmp_path):
    from orion_core.constants import CONFIG_DIR

    assert Path(CONFIG_DIR) not in vs.SETTINGS_PATH.parents      # conftest points it at a temp dir


def test_every_candidate_is_a_real_voice():
    assert {v for v, _ in vs.KOKORO_CANDIDATES} <= set(vs.KOKORO_VOICES)
    assert {v for v, _ in vs.CLOUD_CANDIDATES} <= set(vs.GEMINI_VOICES)
    assert vs.KOKORO_CANDIDATES[0][0] == vs.KOKORO_DEFAULT_VOICE      # today's voice leads
    assert vs.CLOUD_CANDIDATES[0][0] == vs.CLOUD_DEFAULT_VOICE
    assert all(v[:3] in {"af_", "am_", "bf_", "bm_"} for v in vs.KOKORO_VOICES)
    for style in vs.STYLES.values():
        assert style.kokoro_voice in vs.KOKORO_VOICES and style.cloud_voice in vs.GEMINI_VOICES
        assert vs.KOKORO_SPEED_LIMITS[0] <= style.kokoro_speed <= vs.KOKORO_SPEED_LIMITS[1]


def test_the_gemini_list_matches_the_phone_speech_list():
    from orion_core import remote_speech

    assert set(vs.GEMINI_VOICES) == set(remote_speech.GEMINI_VOICES)
    assert len(vs.GEMINI_VOICES) == len(set(vs.GEMINI_VOICES)) == 30


def test_candidate_labels_are_title_case():
    from orion_core.gui.labels import title_case

    for _value, label in (*vs.KOKORO_CANDIDATES, *vs.CLOUD_CANDIDATES):
        assert title_case(label) == label, label


# ── storing, normalising, overriding ─────────────────────────────────────────

@pytest.mark.parametrize("said,voice", [
    ("bm_lewis", "bm_lewis"), ("Lewis", "bm_lewis"), ("BM-Daniel", "bm_daniel"), ("puck", "am_puck"),
    ("George", "bm_george"), ("  liam ", "am_liam"), ("bm_lewis", "bm_lewis"),
])
def test_kokoro_voices_are_understood(said, voice):
    assert vs.normalise_kokoro_voice(said) == voice


@pytest.mark.parametrize("said", ["", None, "Zelda", "xx_heart", "jf_alpha", "zf_xiaobei", "af_"])
def test_other_names_are_not_kokoro_voices(said):
    assert vs.normalise_kokoro_voice(said) is None


def test_gemini_voices_are_case_insensitive_and_exact():
    assert vs.normalise_cloud_voice("orus") == "Orus" and vs.normalise_cloud_voice("CHARON ") == "Charon"
    assert vs.normalise_cloud_voice("Charon\ninjected") is None
    assert vs.normalise_cloud_voice("Chiron") is None and vs.normalise_cloud_voice("") is None


def test_speed_is_clamped_and_must_be_a_number():
    assert vs.normalise_speed(1.1) == 1.1 and vs.normalise_speed("1.05") == 1.05
    assert vs.normalise_speed(5) == 1.25 and vs.normalise_speed(0.1) == 0.8
    for bad in ("fast", None, True, float("nan"), [1]):
        assert vs.normalise_speed(bad) is None


def test_a_choice_is_stored_and_survives_a_fresh_read():
    assert vs.set_value("kokoro_voice", "lewis") and vs.set_value("cloud_voice", "orus")
    assert vs.set_value("kokoro_speed", 1.12)
    stored = json.loads(vs.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert stored["kokoro_voice"] == "bm_lewis" and stored["cloud_voice"] == "Orus"
    assert stored["kokoro_speed"] == 1.12
    assert vs.snapshot() == {"kokoro_voice": "bm_lewis", "kokoro_speed": 1.12, "cloud_voice": "Orus",
                             "reply_pause": "natural", "expressive": True}
    assert vs.kokoro_voice() == "bm_lewis" and vs.cloud_voice() == "Orus" and vs.kokoro_speed() == 1.12


@pytest.mark.parametrize("key,value", [
    ("kokoro_voice", "Zelda"), ("cloud_voice", "Chiron"), ("kokoro_speed", "fast"),
    ("nonsense", "x"), ("cloud_voice", ""),
])
def test_a_value_that_is_no_real_voice_is_never_stored(key, value):
    assert vs.set_value(key, value) is False
    assert not vs.SETTINGS_PATH.exists()
    assert vs.snapshot() == vs.DEFAULTS


def test_a_damaged_file_or_a_bad_stored_value_falls_back_to_the_default():
    vs.SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    vs.SETTINGS_PATH.write_text("{ not json", encoding="utf-8")
    assert vs.snapshot() == vs.DEFAULTS
    vs.SETTINGS_PATH.write_text(json.dumps({"kokoro_voice": "Zelda", "cloud_voice": 7,
                                            "kokoro_speed": "x"}), encoding="utf-8")
    assert vs.snapshot() == vs.DEFAULTS


def test_the_environment_beats_the_stored_choice_for_one_run(monkeypatch):
    vs.set_value("cloud_voice", "Orus")
    monkeypatch.setenv("ORION_VOICE_CLOUD_VOICE", "algenib")
    monkeypatch.setenv("ORION_VOICE_KOKORO_VOICE", "am_michael")
    monkeypatch.setenv("ORION_VOICE_KOKORO_SPEED", "1.2")
    assert vs.cloud_voice() == "Algenib" and vs.kokoro_voice() == "am_michael" and vs.kokoro_speed() == 1.2
    assert vs.stored("cloud_voice") == "Orus"                        # the file is untouched
    monkeypatch.setenv("ORION_VOICE_CLOUD_VOICE", "banana")          # a bad override is ignored
    assert vs.cloud_voice() == "Orus"
    monkeypatch.setenv("ORION_VOICE_CLOUD_VOICE", "leda")            # so is a woman's voice
    assert vs.cloud_voice() == "Orus"


def test_reset_returns_to_charon_and_george():
    vs.set_value("cloud_voice", "Orus")
    assert vs.reset() and vs.snapshot() == vs.DEFAULTS


# ── styles ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,style", [
    ("use your cute voice", ""), ("sound cuter", ""),       # the cute style was removed
    ("the cute girl voice", ""), ("a cute female voice", ""),
    ("use your professional voice", "classic"), ("be formal again", "classic"),
    ("go back to your normal voice", "classic"), ("your default voice", "classic"),
    ("use the old voice", "classic"), ("charon", "classic"), ("an execute voice", ""),
])
def test_style_words(phrase, style):
    assert vs.style_from_words(phrase) == style


def test_a_style_changes_all_three_and_is_recognised_again():
    # Mark 39 retired the cute style; Mark XXXX.4 removed it: one voice, one manner.
    message, ok = vs.apply_style("cute")
    assert not ok and "one way, in my own voice" in message
    assert vs.snapshot() == vs.DEFAULTS and vs.chosen_style() == "classic" and not vs.manner()
    message, ok = vs.apply_style("use your cute girl voice")
    assert not ok and "own voice" in message and vs.chosen_style() == "classic"
    assert not vs.apply_style("cute_male")[1]                                  # the old name
    assert vs.apply_style("professional")[1] and vs.snapshot() == vs.DEFAULTS
    assert vs.chosen_style() == "classic" and vs.STYLES["classic"].label == "Professional"
    assert not vs.manner()
    message, ok = vs.apply_style("spooky")
    assert not ok and "professional" in message and vs.snapshot() == vs.DEFAULTS


def test_describe_names_the_voices():
    vs.apply_style("professional")
    text = vs.describe()
    assert "Charon" in text and "bm_george" in text and "style professional" in text


# ── the offline voice reads it ───────────────────────────────────────────────

def test_kokoro_follows_the_setting_and_an_explicit_file_stays_a_seam(tmp_path):
    vs.set_value("kokoro_voice", "bm_lewis")
    vs.set_value("kokoro_speed", 1.12)
    assert vk.current_voice() == "bm_lewis" and vk.base_speed() == 1.12
    assert vk.KokoroVoice().voice == "bm_lewis"
    legacy = tmp_path / "kokoro.json"
    legacy.write_text(json.dumps({"voice": "lewis", "speed": 0.9}), encoding="utf-8")
    assert vk.current_voice(legacy) == "bm_lewis" and vk.base_speed(legacy) == 0.9
    assert vk.KokoroVoice(settings=legacy).voice == "bm_lewis"


def test_the_older_kokoro_json_still_counts_when_nothing_is_stored(tmp_path, monkeypatch):
    legacy = tmp_path / "voices" / "kokoro.json"
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({"voice": "daniel", "speed": 1.2}), encoding="utf-8")
    monkeypatch.setattr(vk, "settings_path", lambda: legacy)
    assert vk.current_voice() == "bm_daniel" and vk.base_speed() == 1.2
    vs.set_value("kokoro_voice", "am_michael")               # the new store wins from then on
    assert vk.current_voice() == "am_michael"


def test_choosing_an_offline_voice_goes_to_the_one_store_and_women_are_allowed():
    message, ok = vk.choose_voice("Lewis")
    assert ok and "bm_lewis" in message and vs.stored("kokoro_voice") == "bm_lewis"
    message, ok = vk.choose_voice("Zelda")
    assert not ok and "bm_lewis" in message and vk.current_voice() == "bm_lewis"
    assert "bm_lewis" in vk.voices_text() and "bm_george" in vk.voices_text()
    assert "American women" in vk.voices_text()


def test_the_synthesiser_uses_the_chosen_voice_speed_and_accent(monkeypatch, tmp_path):
    import numpy as np

    calls = []

    class Engine:
        voices = None

        def create(self, text, voice, speed, lang):
            calls.append((voice, speed, lang))
            return np.full(2400, 0.2, dtype=np.float32), 24000

    monkeypatch.setattr(vk, "enabled", lambda: True)
    monkeypatch.setattr(vk, "model_dir", lambda dirs=None, spec=None: tmp_path)
    voice = vk.KokoroVoice(engine_loader=lambda folder: Engine())

    assert voice.stream_pcm("Hello there.", on_chunk=lambda chunk: None)
    assert calls[-1] == ("bm_george", 1.0, "en-gb")                    # the unchanged default

    vs.apply_style("professional")                           # same voice, his manner only
    assert voice.stream_pcm("Hello there.", on_chunk=lambda chunk: None)
    assert calls[-1] == ("bm_george", 1.0, "en-gb")

    vs.set_value("kokoro_voice", "am_michael")
    assert voice.stream_pcm("Hello there.", on_chunk=lambda chunk: None)
    assert calls[-1][0] == "am_michael" and calls[-1][2] == "en-us"

    vs.set_value("kokoro_voice", "bm_daniel")
    assert voice.stream_pcm("Hello there.", on_chunk=lambda chunk: None)
    assert calls[-1][0] == "bm_daniel" and calls[-1][2] == "en-gb"


# ── the cloud voice reads it ─────────────────────────────────────────────────

def test_charon_and_the_phone_follow_the_cloud_voice():
    from orion_core import remote_speech, voice_charon

    charon = voice_charon.CharonVoice()
    assert charon.voice_name == "Charon" and remote_speech.speech_voice(None) == "Charon"
    vs.set_value("cloud_voice", "Orus")
    assert charon.voice_name == "Orus" and remote_speech.speech_voice(None) == "Orus"
    assert remote_speech.speech_voice("Puck") == "Puck"              # an explicit request still wins
    # Charon's offline cache is keyed by the voice, so a new voice never plays old lines.
    assert voice_charon._cache_key("Hello", "Charon") != voice_charon._cache_key("Hello", "Orus")
    assert charon._cache_path("Hello").name == voice_charon._cache_key("Hello", "Orus") + ".pcm"


def test_the_live_config_takes_its_voice_from_the_setting():
    source = (ROOT / "orion_core" / "live_worker.py").read_text(encoding="utf-8")
    start = source.index("from .voice_settings import cloud_voice as _cloud_voice")
    assert "voice_name = _cloud_voice()" in source[start:start + 200]
    assert "PrebuiltVoiceConfig(voice_name=voice_name)" in source
    assert "except Exception:\n            voice_name = VOICE_PROFILE.gemini_voice_name" in source


# ── asking for it: tool and voice ────────────────────────────────────────────

class _Signal:
    def emit(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        return _Signal()


def _dispatcher():
    from orion_core.dispatcher import OrionDispatcher

    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _Bus()
    return d


def test_the_tool_changes_the_style_and_each_half(monkeypatch, tmp_path):
    from orion_core import voice_elevenlabs as ev

    monkeypatch.setattr(ev, "API_CONFIG_PATH", tmp_path / "api_keys.json")
    d = _dispatcher()
    result = d.elevenlabs_voice_tool({"action": "voice_style", "style": "cute"})
    assert not result.ok and vs.chosen_style() == "classic"
    assert d.elevenlabs_voice_tool({"action": "voice_style", "style": "professional"}).ok
    assert d.elevenlabs_voice_tool({"action": "voice_style"}).text.startswith("Voice:")
    result = d.elevenlabs_voice_tool({"action": "online_voice", "voice": "iapetus"})
    assert result.ok and vs.cloud_voice() == "Iapetus"
    refused = d.elevenlabs_voice_tool({"action": "online_voice", "voice": "Chiron"})
    assert not refused.ok and "Charon" in refused.text and vs.cloud_voice() == "Iapetus"
    assert d.elevenlabs_voice_tool({"action": "voice_speed", "speed": 1.2}).ok and vs.kokoro_speed() == 1.2
    assert not d.elevenlabs_voice_tool({"action": "voice_speed", "speed": "warp"}).ok
    assert not d.elevenlabs_voice_tool({"action": "voice_style", "style": "spooky"}).ok
    status = d.elevenlabs_voice_tool({"action": "status"})
    assert "Voice: online Iapetus" in status.text
    assert d.elevenlabs_voice_tool({"action": "offline_voice", "voice": "Liam"}).ok
    assert vk.current_voice() == "am_liam"
    assert not d.elevenlabs_voice_tool({"action": "offline_voice", "voice": "Bella"}).ok  # never a woman's
    assert not d.elevenlabs_voice_tool({"action": "online_voice", "voice": "Leda"}).ok
    assert vk.current_voice() == "am_liam" and vs.cloud_voice() == "Iapetus"


@pytest.mark.parametrize("phrase,style", [
    ("please go back to your normal voice", "classic"),
    ("sound normal again", "classic"), ("switch to the classic voice", "classic"),
    ("use your professional voice", "classic"), ("sound professional again", "classic"),
])
def test_voice_phrases_reach_the_tool(phrase, style):
    from orion_core.reflex import match_reflex

    match = match_reflex(phrase)
    assert match is not None, phrase
    assert match.tool == "elevenlabs_voice" and match.args["action"] == "voice_style"
    # whatever the reflex hands over, the style it means is what gets applied
    wanted = vs.style_from_words(match.args["style"]) if match.args["style"] not in vs.STYLES \
        else match.args["style"]
    assert wanted == style, (phrase, match.args)


@pytest.mark.parametrize("phrase", ["change your voice", "what voice are you using", "use your voice",
                                    "sound the alarm", "mute your voice"])
def test_the_voice_reflex_leaves_other_sentences_alone(phrase):
    from orion_core.reflex import match_reflex

    match = match_reflex(phrase)
    assert match is None or match.args.get("action") != "voice_style"


def test_the_schema_offers_the_new_voice_actions():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "elevenlabs_voice")
    properties = tool["parameters"]["properties"]
    for action in ("voice_style", "online_voice", "voice_speed"):
        assert action in properties["action"]["description"]
    assert "style" in properties and "speed" in properties
    assert "OFFLINE voice" in tool["description"] and "bm_george" in tool["description"]


# ── Preferences > Voice ──────────────────────────────────────────────────────

_APP = None


@pytest.fixture
def selector():
    from PyQt6.QtWidgets import QApplication

    # Held: an application nobody references is collected at once, and the
    # widgets built next take the whole test run down (exit 127) whenever
    # this file runs on its own.
    global _APP
    _APP = QApplication.instance() or QApplication([])
    from orion_core.gui.preferences import _VoiceSelector

    box = _VoiceSelector()
    yield box
    box.deleteLater()


def test_the_voice_selector_shows_the_defaults_and_the_candidates(selector):
    assert selector.title() == "Voice"
    assert selector.style.currentData() == "classic"
    assert selector.online.currentData() == "Charon" and selector.offline.currentData() == "bm_george"
    assert [selector.offline.itemData(i) for i in range(selector.offline.count())] == [
        v for v, _ in vs.KOKORO_CANDIDATES]
    assert [selector.online.itemData(i) for i in range(selector.online.count())] == [
        v for v, _ in vs.CLOUD_CANDIDATES]
    assert selector.speed.value() == 1.0
    for widget in (selector.style, selector.online, selector.offline, selector.speed):
        assert widget.accessibleName() and widget.toolTip()


def test_choosing_in_the_selector_saves_at_once(selector):
    seen = []
    selector.changed.connect(lambda key, value: seen.append((key, value)))
    selector.offline.setCurrentIndex(selector.offline.findData("bm_daniel"))
    selector.offline.activated.emit(selector.offline.currentIndex())
    assert vs.stored("kokoro_voice") == "bm_daniel" and seen[-1] == ("kokoro_voice", "bm_daniel")
    assert selector.style.currentData() == "custom"
    selector.online.setCurrentIndex(selector.online.findData("Orus"))
    selector.online.activated.emit(selector.online.currentIndex())
    assert vs.stored("cloud_voice") == "Orus"
    selector.speed.setValue(1.15)
    assert vs.stored("kokoro_speed") == 1.15


def test_choosing_a_style_in_the_selector_sets_all_three(selector):
    assert selector.style.findData("cute") == -1             # retired in Mark 39
    vs.set_value("cloud_voice", "Orus")
    selector.refresh()
    selector.style.setCurrentIndex(selector.style.findData("classic"))
    selector.style.activated.emit(selector.style.currentIndex())
    assert vs.snapshot() == vs.DEFAULTS                      # his one voice
    assert selector.offline.currentData() == "bm_george" and selector.online.currentData() == "Charon"
    assert selector.speed.value() == pytest.approx(1.0)
    assert selector.style.currentData() == "classic"


def test_a_failed_save_is_reported_and_the_box_goes_back(selector, monkeypatch):
    monkeypatch.setattr(vs, "set_value", lambda key, value: False)
    selector.offline.setCurrentIndex(selector.offline.findData("am_michael"))
    selector.offline.activated.emit(selector.offline.currentIndex())
    assert "could not be saved" in selector.status.text()
    assert selector.offline.currentData() == "bm_george"


def test_an_environment_override_is_shown(selector, monkeypatch):
    monkeypatch.setenv("ORION_VOICE_CLOUD_VOICE", "Iapetus")
    selector.refresh()
    assert selector.online.currentData() == "Iapetus" and "ORION_VOICE_CLOUD_VOICE" in selector.status.text()


def test_the_preferences_dialog_carries_the_voice_group():
    from PyQt6.QtWidgets import QApplication

    globals().setdefault("_QT_APP", QApplication.instance() or QApplication([]))  # held, or it is collected
    from orion_core.gui.preferences import PreferencesDialog

    dialog = PreferencesDialog(assistant_name="Orion", user_name="")
    try:
        assert dialog.voice.title() == "Voice"
    finally:
        dialog.deleteLater()


def test_the_reply_pause_and_the_expressive_voice_are_settings(tmp_path, monkeypatch):
    from orion_core import voice_settings as vs
    monkeypatch.setattr(vs, "SETTINGS_PATH", tmp_path / "voice_settings.json")
    for name in ("ORION_REPLY_PAUSE", "ORION_EXPRESSIVE_VOICE"):
        monkeypatch.delenv(name, raising=False)
    assert vs.reply_pause() == "natural" and vs.reply_pause_ms() == 1200
    assert vs.expressive() is True
    assert vs.normalise_reply_pause("give me more time") == "patient"
    assert vs.normalise_reply_pause("a bit shorter") == "brisk"
    assert vs.normalise_reply_pause("normal") == "natural"
    assert vs.normalise_reply_pause("purple") is None
    assert vs.set_value("reply_pause", "longer") and vs.reply_pause_ms() == 1700
    assert vs.set_value("expressive", "off") and vs.expressive() is False
    assert vs.reset() and vs.reply_pause() == "natural" and vs.expressive() is True
