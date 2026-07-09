"""Speech-to-text transcription using faster-whisper."""

import logging
from typing import TypedDict, Generator

from .config import DEFAULT_MODEL

logger = logging.getLogger("vid2txt")


class Segment(TypedDict):
    """A transcribed segment with timestamp and text."""
    start: float
    end: float
    text: str


class TranscriptionResult(TypedDict):
    """Complete transcription result."""
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float


class Transcriber:
    """Transcribe audio to text using faster-whisper."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "cuda",
        compute_type: str = "auto",
        model_path: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_path = model_path  # local dir path or None (use HF cache)
        self._model = None  # lazy-loaded

    def _load_model(self) -> None:
        """Load the Whisper model (downloads on first run)."""
        # Resolve device
        if self.device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        elif self.device == "cuda":
            # Verify CUDA is actually available; fall back to CPU if not
            from ctranslate2 import get_cuda_device_count
            if get_cuda_device_count() == 0:
                logger.warning("CUDA not available, falling back to CPU.")
                device = "cpu"
            else:
                device = "cuda"
        else:
            device = self.device

        # Resolve compute type
        if self.compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            compute_type = self.compute_type

        logger.info("Loading Whisper model '%s' on %s (compute: %s)...",
                     self.model_size, device, compute_type)
        logger.info("(First run will download the model — this may take a few minutes)")

        from faster_whisper import WhisperModel

        model_arg = self.model_path if self.model_path else self.model_size
        self._model = WhisperModel(
            model_arg,
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
                ))

        logger.info("Detected language: %s (confidence: %.2f%%)",
                     info.language, info.language_probability * 100)
        logger.info("Transcribed %d segments, %.1f seconds of audio.",
                     len(segments), info.duration)

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
        After the generator is exhausted, language / duration metadata is
        available via these instance attributes:

        - ``_detected_language``
        - ``_language_probability``
        - ``_audio_duration``

        The existing :meth:`transcribe` method (batch) is unchanged.
        """
        if self._model is None:
            self._load_model()

        logger.info("Transcribing audio (streaming mode)...")
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
        )

        self._detected_language = info.language
        self._language_probability = info.language_probability
        self._audio_duration = info.duration

        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                yield Segment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                )

        logger.info("Streaming transcription complete — %s (%.1f%%)",
                     self._detected_language, self._language_probability * 100)
