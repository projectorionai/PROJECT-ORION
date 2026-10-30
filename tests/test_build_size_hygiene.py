"""Lean packaging must retain the native inference paths ORION actually uses."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_inference_without_the_builds_excluded_training_packages(tmp_path):
    """Exercise real inference with training imports refused, without downloads.

    This catches an upstream package making a formerly optional training
    dependency mandatory. It also protects MediaPipe's matplotlib dependency,
    which is easy to mistake for unused development-only plotting code.
    """
    for module in ("faster_whisper", "rapidocr_onnxruntime", "mediapipe",
                   "google.genai", "sklearn", "tokenizers"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"optional inference integration requires {module}")

    script = textwrap.dedent("""\
        import importlib.abc
        import socket
        import sys
        import build_standalone

        excluded = frozenset(build_standalone.EXCLUDES)

        class WithoutTrainingPackages(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in excluded:
                    raise ModuleNotFoundError(
                        'excluded from standalone build: ' + fullname, name=fullname)

        sys.meta_path.insert(0, WithoutTrainingPackages())

        def no_network(*args, **kwargs):
            raise AssertionError('inference smoke check attempted network access')

        socket.socket.connect = no_network
        socket.socket.connect_ex = no_network
        socket.create_connection = no_network

        import numpy as np
        import mediapipe as mp
        from google.genai import types
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from sklearn.ensemble import RandomForestClassifier
        from faster_whisper import WhisperModel
        from faster_whisper.vad import get_speech_timestamps
        from rapidocr_onnxruntime import RapidOCR

        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.zeros((16, 16, 3), dtype=np.uint8))
        assert image.width == 16
        assert types.FunctionDeclaration(name='find_file').name == 'find_file'
        tokenizer = Tokenizer(WordLevel({'[UNK]': 0, 'file': 1}, unk_token='[UNK]'))
        assert tokenizer.encode('file').ids == [1]
        classifier = RandomForestClassifier(n_estimators=1, random_state=0)
        classifier.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        assert classifier.predict([[0]]).shape == (1,)
        assert WhisperModel.__name__ == 'WhisperModel'
        assert get_speech_timestamps(np.zeros(16000, dtype=np.float32)) == []
        ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        result, _ = ocr(np.zeros((64, 64, 3), dtype=np.uint8))
        assert result is None or result == []
        assert not excluded.intersection(sys.modules)
        print('native inference works without training packages')
        """)
    environment = dict(os.environ, ORION_CONFIG_DIR=str(tmp_path / "config"),
                       MPLCONFIGDIR=str(tmp_path / "matplotlib"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    environment.pop("ORION_BUILD_CHESS_TRAINING", None)
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            env=environment, text=True, capture_output=True,
                            timeout=60,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native inference works without training packages" in result.stdout
