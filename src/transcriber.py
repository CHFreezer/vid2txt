"""Speech-to-text transcription using faster-whisper."""

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict, Generator

from .config import DEFAULT_MODEL, SUPPORTED_MODELS, REQUIRED_MODEL_FILES, DEFAULT_WHISPER_MODEL_DIR

logger = logging.getLogger("vid2txt")


class ModelNotFoundError(RuntimeError):
    """Raised when the Whisper model is not found at the configured path."""


class Segment(TypedDict):
    """A transcribed segment with timestamp and text.

    ``translated_text`` is set when translation has been applied,
    otherwise it is ``None``.
    """
    start: float
    end: float
    text: str
    translated_text: str | None


class TranscriptionResult(TypedDict):
    """Complete transcription result."""
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float


@dataclass
class TranscriptionInfo:
    """Metadata produced alongside a streaming transcription.

    Available on :attr:`Transcriber.info` after :meth:`Transcriber.transcribe_stream`
    has been exhausted.
    """
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0


def _model_dir(base: str, size: str) -> Path:
    """Return the directory where model *size* should live under *base*."""
    return Path(base) / f"faster-whisper-{size}"


def is_model_downloaded(base: str, size: str) -> bool:
    """Check whether model *size* is fully downloaded under *base*."""
    md = _model_dir(base, size)
    return all((md / f).exists() for f in REQUIRED_MODEL_FILES)


class Transcriber:
    """Transcribe audio to text using faster-whisper.

    Models are expected under ``<whisper_model_path>/faster-whisper-<model_size>/``.
    If the model directory does not exist or is incomplete, a
    :class:`ModelNotFoundError` is raised — the caller must download the
    model first (e.g. via :func:`src.model_manager.download_model`).
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "cuda",
        compute_type: str = "auto",
        whisper_model_path: str = DEFAULT_WHISPER_MODEL_DIR,
    ) -> None:
        self.model_size = model_size
        if model_size not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model size '{model_size}'. "
                f"Choose from: {', '.join(SUPPORTED_MODELS)}"
            )
        self.device = device
        self.compute_type = compute_type
        self.whisper_model_path = whisper_model_path  # base directory
        self._model = None  # lazy-loaded
        self._stream_info: TranscriptionInfo | None = None

    @property
    def info(self) -> TranscriptionInfo | None:
        """Metadata from the last :meth:`transcribe_stream` call, or ``None``."""
        return self._stream_info

    @property
    def model_dir(self) -> str:
        """Absolute path to this instance's model directory."""
        return str(_model_dir(self.whisper_model_path, self.model_size).resolve())

    def _load_model(self) -> None:
        """Load the Whisper model.  Raises :class:`ModelNotFoundError` if the
        model has not been downloaded to :attr:`model_dir` yet."""
        # --- Verify model exists ---
        if not is_model_downloaded(self.whisper_model_path, self.model_size):
            raise ModelNotFoundError(
                f"Model '{self.model_size}' not found at {self.model_dir}.\n"
                f"Download it first:\n"
                f"  WebUI → select model → click '⬇ 下载模型'\n"
                f"  CLI   → python -c \"from src.model_manager import "
                f"download_model; download_model('{self.model_size}', "
                f"'{self.whisper_model_path}')\""
            )

        # --- Resolve device ---
        if self.device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        elif self.device == "cuda":
            from ctranslate2 import get_cuda_device_count
            if get_cuda_device_count() == 0:
                logger.warning("CUDA not available, falling back to CPU.")
                device = "cpu"
            else:
                device = "cuda"
        else:
            device = self.device

        # --- Resolve compute type ---
        if self.compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            compute_type = self.compute_type

        logger.info("Loading Whisper model '%s' from %s on %s (compute: %s)...",
                     self.model_size, self.model_dir, device, compute_type)

        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_dir,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe *audio_path* and return segments with timestamps."""
        if self._model is None:
            self._load_model()

        logger.info("Transcribing audio...")
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
        )

        segments: list[Segment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append(Segment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    translated_text=None,
                ))

        logger.info("Detected language: %s (confidence: %.2f%%)",
                     info.language, info.language_probability * 100)
        logger.info("Transcribed %d segments, %.1f seconds of audio.",
                     len(segments), info.duration)

        # Keep .info consistent so callers can use it regardless of which
        # transcribe variant was called.
        self._stream_info = TranscriptionInfo(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

    def transcribe_stream(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> Generator[Segment, None, None]:
        """Transcribe *audio_path*, yielding segments as they are recognised.

        This is the streaming variant — segments appear one-by-one as the
        model produces them, making it suitable for WebUI live preview.
        After the generator is exhausted, metadata is available via
        :attr:`Transcriber.info` as a :class:`TranscriptionInfo` with
        ``language``, ``language_probability``, and ``duration`` fields.

        The existing :meth:`transcribe` method (batch) is unchanged.
        """
        if self._model is None:
            self._load_model()

        logger.info("Transcribing audio (streaming mode)...")
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
        )

        self._stream_info = TranscriptionInfo(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                yield Segment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    translated_text=None,
                )

        logger.info("Streaming transcription complete — %s (%.1f%%)",
                     self._stream_info.language,
                     self._stream_info.language_probability * 100)
