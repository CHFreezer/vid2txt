"""Translation using M2M100 via CTranslate2.

Usage::

    translator = Translator(model_path="/path/to/model", device="cpu")
    result = translator.translate("Hello world", source_lang="en", target_lang="zh")
"""

import logging
import os
from typing import Generator

from .transcriber import Segment

logger = logging.getLogger("vid2txt")


class TranslationModelNotFoundError(RuntimeError):
    """Raised when the CTranslate2 model files are not found."""


class Translator:
    """Translate text segments using M2M100 via CTranslate2.

    The model is loaded lazily on first use.  Call :meth:`unload` to free
    GPU memory between pipeline stages.
    """

    def __init__(
        self,
        model_path: str = "./models/m2m100",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        """Load the CTranslate2 model and HF tokenizer."""
        model_dir = os.path.abspath(self._model_path)
        if not os.path.isdir(model_dir) or not os.path.isfile(
            os.path.join(model_dir, "model.bin")
        ):
            raise TranslationModelNotFoundError(
                f"Translation model not found at {model_dir}\n"
                f"Download it first via WebUI or CLI."
            )

        logger.info(
            "Loading translation model from %s (device=%s, compute=%s)...",
            model_dir, self._device, self._compute_type,
        )

        import ctranslate2
        from transformers import AutoTokenizer

        self._model = ctranslate2.Translator(
            model_dir,
            device=self._device,
            compute_type=self._compute_type,
        )

        # Load tokenizer (auto-downloads + caches from HF)
        self._tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
        logger.info("Translation model loaded.")

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        logger.info("Translation model unloaded.")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text string."""
        if self._model is None:
            self._load_model()

        self._tokenizer.src_lang = source_lang
        self._tokenizer.tgt_lang = target_lang
        encoded = self._tokenizer(text, truncation=True)

        # CTranslate2 translate_batch expects list of string token lists
        source_tokens = self._tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        # M2M100 needs target language token as decoder prefix
        tgt_token = f"__{target_lang}__"
        results = self._model.translate_batch(
            [source_tokens],
            target_prefix=[[tgt_token]],
            beam_size=1,
        )

        output_tokens = results[0].hypotheses[0]
        result = self._tokenizer.decode(
            self._tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )
        return result.strip()

    def translate_segments(
        self,
        segments: list[Segment],
        source_lang: str,
        target_lang: str,
    ) -> list[Segment]:
        """Translate all segments, preserving timestamps."""
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
        """Yield segments one at a time as they are translated."""
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
