"""
ORION's own manual: how he works and how to fix what goes wrong, in plain
words, so "why can't you hear me?" or "how do I flash the robot?" is answered
from how he is actually built (owner, 28 Oct 2026: everything TORVIS has -
TORVIS ships "the assistant's own manual").

Kept in the code, not in docs/, so the installed app always has the manual
that matches it.
"""

from __future__ import annotations

import re

#: (title, the words people use for it, the answer)
SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("He can't hear me, or hears the wrong words",
     "hear hearing mic microphone misheard mishears deaf quiet listen listening wrong words",
     "1. Check the microphone in Preferences > Audio is the one you are speaking into, and "
     "that he is not muted (the mic button, or say 'unmute'). 2. Look at the Live Voice line "
     "on the BRAIN page: 'Waiting for connection' means the cloud voice is reconnecting - he "
     "falls back to his offline hearing, which runs on the graphics card. 3. When the cloud "
     "voice misses something his own recogniser heard, he answers it anyway within six "
     "seconds, notes how much of your audio reached the cloud, and refreshes the voice "
     "channel so the next request is heard. Ask 'why did you stop hearing me?' and he reads "
     "his own logs for the cause."),
    ("He is slow, or Windows says not responding",
     "slow lag lagging freeze frozen not responding hang stuck performance fast",
     "Open Diagnostics > Load: the reactor gauge shows CPU, GPU, RAM and network. Ask 'why "
     "are you slow?' and the Reliability agent reads his latency log. Heavy pages (Mission, "
     "Globe, Plugins) open faster since 40.5; a page that keeps stalling is worth a report "
     "('report a bug: ...'). Restarting him ('restart yourself') clears a stuck channel."),
    ("What runs on the graphics card",
     "gpu graphics card cuda directml nvidia rtx vram accelerate",
     "Speech recognition (Whisper) runs on the NVIDIA GPU through CUDA; reading the screen "
     "(OCR), object detection and memory indexing run on it through DirectML. A few small "
     "models stay on the CPU because they are measured faster there. Thinking with Gemini "
     "happens in Google's cloud; the local brain (Ollama) runs on the GPU. ORION_GPU=0 keeps "
     "everything on the CPU. CUDA's cuBLAS lives in %USERPROFILE%\\ORION\\gpu-runtime."),
    ("Driving a robot",
     "robot rover drive motors wheels servo estop emergency arduino esp32 serial usb",
     "Say 'connect to the software robot' to try it with no hardware. A real robot runs "
     "the OrionLink library (robot/orion_link); connect with 'serial:COM5' over USB (needs "
     "the pyserial package) or 'tcp:host:port' on the network. 'Stop' and 'emergency stop' "
     "always work; motion only happens when you ask. The robot stops by itself within half a "
     "second if it stops hearing him."),
    ("Writing and flashing robot firmware",
     "firmware flash upload compile sketch board arduino esp32 code",
     "Ask him to write firmware for your board: he writes the C++, compiles it with every "
     "warning on and shows you the compiler's words if it fails. 'List my robot boards' shows "
     "what is plugged in; flashing happens only when you ask, to the port you name. Arduino "
     "CLI and the board packages live in %USERPROFILE%\\ORION\\toolchain."),
    ("Your data and privacy",
     "privacy data gdpr erase delete everything personal where stored keys secure",
     "Everything he keeps is on this PC, in %USERPROFILE%\\ORION\\config; his keys are sealed "
     "to your Windows account. Your live voice goes to Google's Gemini service (on a paid key "
     "Google does not train on it). 'Erase my data' shows what would go and asks on screen; "
     "once approved, he restarts and sends it to the Recycle Bin. The public source export "
     "refuses anything personal this PC knows about you."),
    ("Leaving the PC on for the phone",
     "away phone remote out leave sleep awake lock stay ready",
     "Say 'away mode on' before you go out: the PC stays awake for your phone and locks itself "
     "after ten minutes without input. 'Away mode off' gives sleep back to Windows."),
    ("The specialist agents",
     "agent agents specialist expert coding research neuroscience security marketing",
     "He hands questions to specialists (coding, research, neuroscience, AI, security, "
     "marketing, reliability, integrations, continuity, files and triage), chosen by what the "
     "request means. With no internet they answer on the local model. Say 'double-check' - or "
     "ask about an exam, essay or coursework - and the answer gets a second, reviewing pass."),
    ("Updates and versions",
     "update version mark release install rollback new changes",
     "He is Mark XXXX (he says 'Mark forty'). 'What's new?' reads his patch notes. Each "
     "release is built, checked, installed with the previous version kept for rollback, and "
     "committed on this PC only."),
    ("Training his local brain",
     "train training local brain hugging face llama fine tune offline model",
     "Create a Hugging Face account, accept the licence on the Llama 3.2 3B Instruct page, "
     "make a read token and run 'hf auth login' in his training environment. Then ask him to "
     "train his local brain; it runs on the GPU."),
)

_WORD = re.compile(r"[a-z0-9]+")


def lookup(question: str, limit: int = 2) -> list[tuple[str, str]]:
    """The sections that best answer *question* (title, answer), best first."""
    asked = set(_WORD.findall(str(question or "").lower()))
    scored = []
    for index, (title, words, answer) in enumerate(SECTIONS):
        keys = set(_WORD.findall(words + " " + title.lower()))
        score = len(asked & keys)
        if score:
            scored.append((-score, index, title, answer))
    scored.sort()
    return [(title, answer) for _s, _i, title, answer in scored[:max(1, limit)]]


def contents() -> str:
    return "Manual sections: " + "; ".join(title for title, _w, _a in SECTIONS) + "."


__all__ = ["SECTIONS", "contents", "lookup"]
