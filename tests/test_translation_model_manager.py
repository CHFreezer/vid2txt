"""Unit tests for translation model manager."""

import os
import tempfile

from src.translation_model_manager import (
    is_model_downloaded,
    get_model_dir,
)


class TestIsModelDownloaded:
    def test_false_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert not is_model_downloaded(tmp)

    def test_false_when_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create only one of the required files
            (tmp_path := os.path.join(tmp, "model.bin"))
            # don't create it, just check
            assert not is_model_downloaded(tmp)

    def test_true_when_all_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            for f in ("model.bin", "config.json", "shared_vocabulary.json"):
                with open(os.path.join(tmp, f), "w") as fh:
                    fh.write("fake")
            assert is_model_downloaded(tmp)


class TestGetModelDir:
    def test_returns_absolute_path(self):
        p = get_model_dir("./models/m2m100")
        assert os.path.isabs(p)
