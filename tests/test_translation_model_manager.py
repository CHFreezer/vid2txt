"""Unit tests for translation model manager."""

import os
import tempfile

from src.translation_model_manager import (
    get_model_path,
    is_model_downloaded,
    list_translation_models,
)


class TestListTranslationModels:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = list_translation_models(tmp)
            assert len(result) == 7  # all 7 models (incl. 1.25Bit)
            for key, info in result.items():
                assert info["downloaded"] is False
                assert os.path.basename(info["path"])  # filename present
                assert info["size_gb"] > 0

    def test_detects_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create a fake GGUF file
            key = "1.8B-Q4_K_M"
            expected_path = get_model_path(key, tmp)
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text("fake")

            result = list_translation_models(tmp)
            assert result[key]["downloaded"] is True
            assert result[key]["path"] == str(expected_path)

    def test_get_model_path(self):
        p = get_model_path("7B-Q4_K_M", "/models")
        assert p.name == "Hy-MT2-7B-Q4_K_M.gguf"


class TestIsModelDownloaded:
    def test_false_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert not is_model_downloaded("1.8B-Q4_K_M", tmp)

    def test_true_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = "1.8B-Q6_K"
            p = get_model_path(key, tmp)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fake")
            assert is_model_downloaded(key, tmp)
