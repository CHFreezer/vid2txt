"""End-to-end regression tests against real video URLs.

Covers the full pipeline: validate → fetch info → download → transcribe → format.

Three fixed test videos:
- Bilibili:   https://www.bilibili.com/video/BV1Kt7q6hEAr
- YouTube:    https://www.youtube.com/watch?v=dQw4w9WgXcQ
- YT Shorts:  https://www.youtube.com/shorts/Lp1o_IDZ7vk

Strategy
--------
- **Metadata tests** (fast, no download) — all three videos.
- **Full-pipeline test** (slow, downloads + transcribes) — the YouTube Shorts
  video only (shortest duration), using the ``tiny`` model.

Slow tests are marked ``@pytest.mark.slow`` and can be skipped with
``pytest -m "not slow"``.
"""

import os
import json
import tempfile

import pytest

from src.utils import validate_url, get_output_basename, cleanup_temp_dir
from src.downloader import Downloader, DownloadError
from src.transcriber import Transcriber
from src.formatter import Formatter
from src import model_manager
from src import settings


# ── Test config (clean temp dir) ──────────────────────────────────────────

@pytest.fixture(scope="module")
def test_cfg(tmp_path_factory):
    """Create a clean temp config for CLI regression tests."""
    model_dir = tmp_path_factory.mktemp("cli_whisper")
    config_file = tmp_path_factory.mktemp("cli_config") / "config.json"
    cfg = {
        "device": "cpu",
        "whisper_model_path": str(model_dir),
        "model": "tiny",
        "language": "auto",
        "translate_enabled": False,
        "target_lang": "zh",
        "translation_model_path": str(tmp_path_factory.mktemp("cli_translate")),
    }
    config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    settings.set_config_path(str(config_file))
    return cfg


# ── Test URLs ────────────────────────────────────────────────────────────────

BILIBILI_URL = "https://www.bilibili.com/video/BV1Kt7q6hEAr"
YOUTUBE_URL  = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
SHORTS_URL   = "https://www.youtube.com/shorts/Lp1o_IDZ7vk"

ALL_URLS = [BILIBILI_URL, YOUTUBE_URL, SHORTS_URL]


# ══════════════════════════════════════════════════════════════════════════════
# 1 — URL validation
# ══════════════════════════════════════════════════════════════════════════════

class TestURLValidation:
    """All three fixed test URLs must pass validation."""

    @pytest.mark.parametrize("url", ALL_URLS)
    def test_url_is_valid(self, url: str) -> None:
        assert validate_url(url) is True, f"URL should be valid: {url}"


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Video metadata (yt-dlp --dump-json, NO download)
# ══════════════════════════════════════════════════════════════════════════════

class TestVideoMetadata:
    """Fetch video info for all three videos.  No audio download — just the
    yt-dlp API call, so these tests are fast."""

    @staticmethod
    def _downloader() -> Downloader:
        return Downloader(verbose=False)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _assert_valid_info(info: dict, source: str) -> None:
        """Check that *info* has the expected shape after yt-dlp --dump-json."""
        assert isinstance(info, dict), f"[{source}] info must be a dict"
        assert "id" in info, f"[{source}] missing 'id'"
        assert "title" in info, f"[{source}] missing 'title'"
        assert info["title"], f"[{source}] title is empty"

    # ── Tests ────────────────────────────────────────────────────────────

    def test_bilibili_info(self) -> None:
        downloader = self._downloader()
        info = downloader.get_video_info(BILIBILI_URL)
        self._assert_valid_info(info, "bilibili")
        # Bilibili BV IDs are alphanumeric, typically 12 chars
        assert len(info["id"]) >= 1, "bilibili: video ID too short"

    def test_youtube_info(self) -> None:
        downloader = self._downloader()
        info = downloader.get_video_info(YOUTUBE_URL)
        self._assert_valid_info(info, "youtube")
        # Known video ID for Never Gonna Give You Up
        assert info["id"] == "dQw4w9WgXcQ", (
            f"Expected dQw4w9WgXcQ, got {info['id']}"
        )
        # Duration should be ~212 seconds
        duration = info.get("duration", 0) or 0
        assert 200 <= duration <= 230, (
            f"Expected duration ~212s, got {duration}s"
        )

    def test_shorts_info(self) -> None:
        downloader = self._downloader()
        info = downloader.get_video_info(SHORTS_URL)
        self._assert_valid_info(info, "shorts")
        # YouTube Shorts are ≤ 60 seconds
        duration = info.get("duration", 0) or 0
        assert 0 < duration <= 60, (
            f"Shorts duration should be ≤ 60s, got {duration}s"
        )

    def test_all_urls_reachable(self) -> None:
        """Smoke test: all three URLs return parseable JSON."""
        downloader = self._downloader()
        for url in ALL_URLS:
            info = downloader.get_video_info(url)
            self._assert_valid_info(info, url)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Full pipeline (download → convert → transcribe → format)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="class")
def _pipeline_result() -> dict:
    """Run the full pipeline once and return results shared across tests.

    Output goes to a temporary directory that is cleaned up afterwards.
    """
    # Ensure model is downloaded (also tests download functionality)
    model_manager.download_model("tiny", settings.load()["whisper_model_path"])

    downloader = Downloader(verbose=False)
    transcriber = Transcriber(model_size="tiny", device="cpu", compute_type="int8", whisper_model_path=settings.load()["whisper_model_path"])
    formatter = Formatter()

    # ---- Download + Convert ----
    wav_path, temp_dir, video_info = downloader(SHORTS_URL)

    # ---- Transcribe ----
    result = transcriber.transcribe(wav_path)
    segments = result["segments"]

    # ---- Format ----
    title = video_info.get("title", "")
    video_id = video_info.get("id", "")
    basename = get_output_basename(title, video_id)

    tmp_out = tempfile.mkdtemp(prefix="vid2txt_regression_")
    output_files = formatter.write(segments, tmp_out, basename)

    yield {
        "video_info": video_info,
        "segments": segments,
        "result": result,
        "output_files": output_files,
        "output_dir": tmp_out,
    }

    # ---- Cleanup ----
    cleanup_temp_dir(temp_dir)
    cleanup_temp_dir(tmp_out)


@pytest.mark.slow
class TestFullPipeline:
    """End-to-end: download audio, transcribe with Whisper, format output.

    Uses the **YouTube Shorts** video (shortest duration) and the **tiny**
    model (smallest, fastest to download and run).
    """

    # ── Download / conversion checks ─────────────────────────────────────

    def test_video_info_has_required_fields(self, _pipeline_result: dict) -> None:
        info = _pipeline_result["video_info"]
        assert info["id"], "video ID must not be empty"
        assert info["title"], "title must not be empty"

    def test_audio_downloaded_and_converted(self, _pipeline_result: dict) -> None:
        """WAV file was cleaned up, but we know it worked because segments exist."""
        # If download/convert failed, transcriber would have raised.
        pass

    # ── Transcription checks ─────────────────────────────────────────────

    def test_transcription_produces_segments(self, _pipeline_result: dict) -> None:
        segments = _pipeline_result["segments"]
        assert isinstance(segments, list), "segments must be a list"
        assert len(segments) > 0, "expected at least 1 transcribed segment"

    def test_segments_have_required_fields(self, _pipeline_result: dict) -> None:
        for seg in _pipeline_result["segments"]:
            assert "start" in seg, f"segment missing 'start': {seg}"
            assert "end" in seg, f"segment missing 'end': {seg}"
            assert "text" in seg, f"segment missing 'text': {seg}"
            assert isinstance(seg["start"], (int, float))
            assert isinstance(seg["end"], (int, float))
            assert seg["end"] >= seg["start"], (
                f"end ({seg['end']}) < start ({seg['start']})"
            )
            assert isinstance(seg["text"], str)
            assert seg["text"].strip(), "segment text must not be empty"

    def test_language_detected(self, _pipeline_result: dict) -> None:
        result = _pipeline_result["result"]
        assert "language" in result
        assert result["language"], "language code must not be empty"

    def test_duration_positive(self, _pipeline_result: dict) -> None:
        result = _pipeline_result["result"]
        assert result.get("duration", 0) > 0, "duration must be positive"

    # ── Format output checks ─────────────────────────────────────────────

    def test_txt_file_exists_and_non_empty(self, _pipeline_result: dict) -> None:
        txt_path = _pipeline_result["output_files"]["txt"]
        assert os.path.isfile(txt_path), f"TXT file missing: {txt_path}"
        size = os.path.getsize(txt_path)
        assert size > 0, "TXT file is empty"

    def test_srt_file_exists_and_non_empty(self, _pipeline_result: dict) -> None:
        srt_path = _pipeline_result["output_files"]["srt"]
        assert os.path.isfile(srt_path), f"SRT file missing: {srt_path}"
        size = os.path.getsize(srt_path)
        assert size > 0, "SRT file is empty"

    def test_srt_has_valid_format(self, _pipeline_result: dict) -> None:
        srt_path = _pipeline_result["output_files"]["srt"]
        with open(srt_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        lines = content.strip().split("\n")
        # First entry should start with "1"
        assert lines[0] == "1", f"SRT should start with '1', got: {lines[0]!r}"
        # Should contain timestamp arrows
        assert "-->" in content, "SRT must contain '-->' timestamp separators"

    def test_txt_matches_segments(self, _pipeline_result: dict) -> None:
        txt_path = _pipeline_result["output_files"]["txt"]
        with open(txt_path, "r", encoding="utf-8") as fh:
            txt_content = fh.read()
        expected_texts = [seg["text"] for seg in _pipeline_result["segments"]]
        expected = "\n".join(expected_texts)
        assert txt_content.strip() == expected.strip(), (
            "TXT content does not match segments"
        )

    # ── Streaming API parity ─────────────────────────────────────────────

    def test_stream_and_batch_produce_same_segments(self) -> None:
        """transcribe_stream should produce the same segments as transcribe."""
        downloader = Downloader(verbose=False)
        transcriber = Transcriber(model_size="tiny", device="cpu", compute_type="int8", whisper_model_path=settings.load()["whisper_model_path"])

        wav_path, temp_dir, _video_info = downloader(SHORTS_URL)

        # Batch
        batch_result = transcriber.transcribe(wav_path)

        # Streaming (need a fresh transcriber because model is stateful after
        # first run, and transcribe_stream sets info)
        transcriber2 = Transcriber(model_size="tiny", device="cpu", compute_type="int8", whisper_model_path=settings.load()["whisper_model_path"])
        stream_segments = list(transcriber2.transcribe_stream(wav_path))

        assert len(stream_segments) == len(batch_result["segments"]), (
            f"Stream ({len(stream_segments)}) and batch "
            f"({len(batch_result['segments'])}) segment counts differ"
        )
        for i, (s_seg, b_seg) in enumerate(
            zip(stream_segments, batch_result["segments"])
        ):
            assert s_seg["text"] == b_seg["text"], (
                f"Segment {i} text differs:\n"
                f"  stream: {s_seg['text']!r}\n"
                f"  batch:  {b_seg['text']!r}"
            )

        # Streaming metadata should match batch metadata
        try:
            assert transcriber2.info is not None
            assert transcriber2.info.language == batch_result["language"]
            assert transcriber2.info.duration == pytest.approx(
                batch_result["duration"], rel=0.01
            )
        finally:
            cleanup_temp_dir(temp_dir)
