# ORION Dependency Health Audit (Phase 8, Section E)

Generated from a live probe of the host Python environment (Python 3.13),
2026-07-05. Grouped as the directive requires: required, optional, deprecated,
and recommended replacements — with exact install commands.

## Required (installed and load cleanly ✅)

Core runtime — ORION will not start without these:

| Package | Role |
|---|---|
| PyQt6 + **PyQt6-WebEngine** | GUI shell **and the 3-D quantum face / globe (WebGL)** |
| qasync | asyncio ⇄ Qt event-loop bridge |
| aiohttp | all async HTTP (providers, briefing, geo, weather) |
| google-genai | Gemini Live native-audio channel |
| sounddevice, numpy | audio capture/playback |
| psutil | telemetry, process governor, sentinel |
| Pillow | image handling for vision/OCR |

All present. No action needed.

## Required for full capability (installed ✅)

| Package | Capability | Note |
|---|---|---|
| pywin32 | Outlook COM, Win32 window focus | Windows only |
| pywinauto | **native UIA text editing** (Notepad fix, Section C) | |
| pyperclip | clipboard-injection editing fallback | |
| pyautogui, keyboard | cursor/keyboard automation | |
| pygetwindow | window enumeration | |
| RapidOCR (rapidocr-onnxruntime) | **primary OCR** — ONNX, offline, per-box confidence | |
| OpenCV (cv2) | OCR preprocessing (Section B) | |
| pypdf | PDF text extraction | |
| python-docx, matplotlib | document export, charts | |
| vosk | **wake-word + voice-interruption listener** | *was missing — installed this phase; en-us model auto-downloads on first run* |
| faster-whisper, openai-whisper | offline speech-to-text | |
| edge-tts | high-quality online TTS fallback | |
| pyttsx3 | offline SAPI voice (default, male, frozen profile) | |
| webrtcvad | voice-activity detection | |

## Optional (not installed — heavy; install only if you want the capability)

| Package | Adds | Install | Cost |
|---|---|---|---|
| **EasyOCR** | 2nd OCR engine in the confidence-fallback chain | `pip install easyocr` | pulls PyTorch (~2 GB) |
| **PaddleOCR** | 3rd OCR engine (very accurate) | `pip install paddleocr paddlepaddle` | large (~1.5 GB) |
| **silero-vad** | neural VAD (better than webrtcvad in noise) | `pip install silero-vad torch` | pulls PyTorch |
| **piper-tts** | neural offline male voice (upgrade from SAPI) | `pip install piper-tts` + a male `.onnx` voice, set `ORION_PIPER_VOICE` | ~60 MB per voice |
| **Coqui TTS (XTTS)** | highest-quality/voice-cloning TTS | `pip install TTS` | heavy, PyTorch |

The OCR chain and voice stack already function fully without these — they are
strict upgrades, gated behind PyTorch's footprint, and ORION degrades to the
installed engines automatically.

## Needs external binary (installed but inert ⚠️)

| Package | Issue | Fix |
|---|---|---|
| **pytesseract** | the Python wrapper is present, but the **Tesseract engine binary** is not on PATH, so it cannot run | Install the Windows build from `https://github.com/UB-Mannheim/tesseract/wiki`, then add its folder to PATH (or set `pytesseract.pytesseract.tesseract_cmd`). Until then RapidOCR/WinRT carry OCR. |

## Deprecated / superseded (in-code, no action forced)

| Was | Now | Status |
|---|---|---|
| Open-Meteo geocoder (globe) | **GeoIntelligenceEngine** (OSM/Nominatim + Overpass) | Open-Meteo kept only as an offline-lite fallback |
| single-engine `OcrEngine` | confidence-scored **multi-engine chain** | old `image_to_text` preserved as a shim |
| memory-based news dedup | dedicated **NewsSignatureCache** (TTL) | old path was never wired; removed |

## Geospatial services (no pip package — HTTP APIs)

GeoNames, Nominatim and Overpass are **web services**, not Python packages, so
nothing to install. ORION calls them directly with a rate limiter and caches
to `config/geo_cache.db`. If you want a fully-offline gazetteer later, the
route is a local GeoNames dump (`allCountries.zip`) loaded into SQLite —
noted as a future enhancement, not a dependency.

## One-line install for the optional upgrades

```powershell
# OCR: add EasyOCR + PaddleOCR to the fallback chain (heavy, PyTorch)
pip install easyocr paddleocr paddlepaddle

# Voice: neural offline TTS + neural VAD (heavy, PyTorch)
pip install piper-tts silero-vad torch

# OCR binary engine (no pip — download the Windows installer)
#   https://github.com/UB-Mannheim/tesseract/wiki  (then add to PATH)
```

## Verdict

No missing **required** dependency; no version conflicts observed on import
(83 modules import cleanly). The only genuine gap closed this phase was
**Vosk** (its absence had silently disabled the wake-word and
voice-interruption listeners). Everything else missing is an optional,
PyTorch-scale upgrade that ORION already works without.
