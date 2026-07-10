"""Translation using Hy-MT2 GGUF models via llama-cpp-python.

Usage::

    translator = Translator(model_path="/path/to/model.gguf", device="cuda")
    result = translator.translate("你好世界", source_lang="zh", target_lang="en")
    # → "Hello World"
"""

import logging
from typing import Generator

from .config import TRANSLATION_INFERENCE_PARAMS
from .transcriber import Segment

logger = logging.getLogger("vid2txt")


class TranslationModelNotFoundError(RuntimeError):
    """Raised when the GGUF translation model file does not exist."""


class Translator:
    """Translate text segments using a Hy-MT2 GGUF model.

    The model is loaded lazily on first use.  Call :meth:`unload` to free
    GPU memory between pipeline stages.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        n_gpu_layers: int = 0,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._n_gpu_layers = n_gpu_layers
        self._model = None  # lazy-loaded

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        """Load the GGUF model.  Raises TranslationModelNotFoundError if the
        file does not exist."""
        import os as _os

        if not _os.path.isfile(self._model_path):
            raise TranslationModelNotFoundError(
                f"Translation model not found: {self._model_path}\n"
                f"Download it first:\n"
                f"  WebUI → 勾选翻译 → 选择模型 → 点击下载\n"
                f"  CLI   → python -c \"from src.translation_model_manager import "
                f"download_translation_model; download_translation_model("
                f"'1.8B-Q4_K_M', 'models/hy-mt2')\""
            )

        logger.info(
            "Loading translation model from %s (device=%s, n_gpu_layers=%d)...",
            self._model_path, self._device, self._n_gpu_layers,
        )

        from llama_cpp import Llama

        self._model = Llama(
            model_path=self._model_path,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=TRANSLATION_INFERENCE_PARAMS["max_tokens"],
            verbose=False,
        )
        logger.info("Translation model loaded.")

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Translation model unloaded.")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text string.

        Args:
            text: Source text to translate.
            source_lang: Source language code (e.g. ``"zh"``).
            target_lang: Target language code (e.g. ``"en"``).

        Returns:
            Translated text string.
        """
        if self._model is None:
            self._load_model()

        prompt = (
            f"Translate from {source_lang} to {target_lang}:\n"
            f"{text}"
        )

        params = dict(TRANSLATION_INFERENCE_PARAMS)
        params.pop("max_tokens")  # use default from model context

        output = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            repeat_penalty=params["repetition_penalty"],
        )

        result = output["choices"][0]["message"]["content"].strip()
        return result

    def translate_segments(
        self,
        segments: list[Segment],
        source_lang: str,
        target_lang: str,
    ) -> list[Segment]:
        """Translate all segments, returning copies with ``translated_text`` set.

        Timestamps (``start`` / ``end``) are preserved unchanged.
        """
        results: list[Segment] = []
        for seg in segments:
            translated = self.translate(
                seg["text"], source_lang=source_lang, target_lang=target_lang
            )
            results.append(Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                translated_text=translated,
            ))
        return results

    def translate_segments_stream(
        self,
        segments: list[Segment],
        source_lang: str,
        target_lang: str,
    ) -> Generator[Segment, None, None]:
        """Yield segments one at a time as they are translated.

        Suitable for WebUI live preview — each segment is yielded as soon
        as its translation completes, so the UI can update progressively.
        """
        total = len(segments)
        for i, seg in enumerate(segments):
            translated = self.translate(
                seg["text"], source_lang=source_lang, target_lang=target_lang
            )
            logger.debug("Translated segment %d/%d", i + 1, total)
            yield Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                translated_text=translated,
            )
