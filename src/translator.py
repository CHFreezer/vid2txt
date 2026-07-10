"""Translation using M2M100 via CTranslate2 — no PyTorch/transformers needed."""

import json
import logging
import os
from typing import Generator

from .transcriber import Segment
from .config import DEFAULT_TRANSLATION_MODEL_DIR

logger = logging.getLogger("vid2txt")


class TranslationModelNotFoundError(RuntimeError):
    """Raised when the CTranslate2 model files are not found."""


class Translator:
    """Translate text segments using M2M100 via CTranslate2.

    Uses sentencepiece for tokenization — no PyTorch / transformers dependency.
    Language tokens (__en__, __zh__, etc.) are resolved from the model's
    shared_vocabulary.json.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_TRANSLATION_MODEL_DIR,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._sp = None
        self._lang_tokens: dict[str, int] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        """Load CTranslate2 model and sentencepiece tokenizer."""
        model_dir = os.path.abspath(self._model_path)
        if not os.path.isdir(model_dir) or not os.path.isfile(
            os.path.join(model_dir, "model.bin")
        ):
            raise TranslationModelNotFoundError(
                f"Translation model not found at {model_dir}\n"
                f"Download it first via WebUI or CLI."
            )

        logger.info("Loading translation model from %s (device=%s, compute=%s)...",
                     model_dir, self._device, self._compute_type)

        import ctranslate2
        import sentencepiece as spm

        self._model = ctranslate2.Translator(
            model_dir, device=self._device, compute_type=self._compute_type,
        )

        # Load sentencepiece tokenizer from model directory
        spm_path = os.path.join(model_dir, "sentencepiece.bpe.model")
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(spm_path)

        # Load language token IDs from shared vocabulary
        vocab_path = os.path.join(model_dir, "shared_vocabulary.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # Build mapping: language code → token string (e.g. "zh" → "__zh__")
        self._lang_tokens = {}
        for token_str in vocab:
            if token_str.startswith("__") and token_str.endswith("__"):
                lang = token_str.strip("_")
                self._lang_tokens[lang] = token_str

        logger.info("Translation model loaded (%d languages).", len(self._lang_tokens))

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._sp is not None:
            del self._sp
            self._sp = None
        self._lang_tokens.clear()
        logger.info("Translation model unloaded.")

    def translate(
        self, text: str, source_lang: str, target_lang: str,
    ) -> str:
        """Translate a single text string.

        Args:
            text: Source text to translate.
            source_lang: Source language code (e.g. ``"en"``).
            target_lang: Target language code (e.g. ``"zh"``).

        Returns:
            Translated text string.
        """
        if self._model is None:
            self._load_model()

        # M2M100: source_lang token as first input token; target_lang as decoder prefix
        src_token = self._lang_tokens.get(source_lang, f"__{source_lang}__")
        tgt_token = self._lang_tokens.get(target_lang, f"__{target_lang}__")

        # Build tokens manually: language token + tokenized text + EOS
        text_tokens = self._sp.encode(text, out_type=str)
        tokens = [src_token] + text_tokens + ["</s>"]

        results = self._model.translate_batch(
            [tokens],
            target_prefix=[[tgt_token]],
            beam_size=1,
        )

        output = results[0].hypotheses[0]
        # Strip leading target language token if present
        if output and output[0] == tgt_token:
            output = output[1:]
        result = self._sp.decode(output).strip()
        return result

    def translate_segments(
        self, segments: list[Segment], source_lang: str, target_lang: str,
    ) -> list[Segment]:
        """Translate all segments, preserving timestamps."""
        results: list[Segment] = []
        for seg in segments:
            translated = self.translate(
                seg["text"], source_lang=source_lang, target_lang=target_lang
            )
            results.append(Segment(
                start=seg["start"], end=seg["end"],
                text=seg["text"], translated_text=translated,
            ))
        return results

    def translate_segments_stream(
        self, segments: list[Segment], source_lang: str, target_lang: str,
    ) -> Generator[Segment, None, None]:
        """Yield segments one at a time as they are translated."""
        total = len(segments)
        for i, seg in enumerate(segments):
            translated = self.translate(
                seg["text"], source_lang=source_lang, target_lang=target_lang
            )
            logger.debug("Translated segment %d/%d", i + 1, total)
            yield Segment(
                start=seg["start"], end=seg["end"],
                text=seg["text"], translated_text=translated,
            )
