"""Unit tests for src.transcriber."""

import tempfile
import os
from pathlib import Path

import pytest

from src.transcriber import Transcriber, ModelNotFoundError


class TestModelNotFound:
    """Verify that Transcriber raises ModelNotFoundError when the model
    directory does not exist or is incomplete."""

    def test_raises_when_model_dir_missing(self) -> None:
        """Transcriber should raise ModelNotFoundError for a non-existent path."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "nonexistent_models")
            t = Transcriber(model_size="tiny", model_path=model_path, device="cpu", compute_type="int8")
            with pytest.raises(ModelNotFoundError):
                t._load_model()

    def test_raises_when_model_dir_empty(self) -> None:
        """Transcriber should raise ModelNotFoundError when the model dir
        exists but contains no model files."""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "faster-whisper-tiny")
            os.makedirs(model_dir, exist_ok=True)
            t = Transcriber(model_size="tiny", model_path=tmp, device="cpu", compute_type="int8")
            with pytest.raises(ModelNotFoundError):
                t._load_model()

    def test_raises_when_model_incomplete(self) -> None:
        """Transcriber should raise when only some required files are present."""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "faster-whisper-tiny")
            os.makedirs(model_dir, exist_ok=True)
            # Create only config.json — missing model.bin, tokenizer.json, vocabulary.txt
            Path(model_dir, "config.json").write_text("{}")
            t = Transcriber(model_size="tiny", model_path=tmp, device="cpu", compute_type="int8")
            with pytest.raises(ModelNotFoundError):
                t._load_model()

    def test_unsupported_model_size(self) -> None:
        """Transcriber should reject unknown model sizes at init time."""
        with pytest.raises(ValueError, match="Unsupported model size"):
            Transcriber(model_size="huge", device="cpu")

    def test_info_is_none_before_transcription(self) -> None:
        """info should be None before any transcription runs."""
        with tempfile.TemporaryDirectory() as tmp:
            t = Transcriber(model_size="tiny", model_path=tmp, device="cpu", compute_type="int8")
            assert t.info is None
