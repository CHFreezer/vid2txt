"""Audio downloader — yt-dlp + ffmpeg pipeline."""

import os
import json
import shutil
import subprocess
import tempfile
import logging
import glob

from .config import SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_CODEC, YT_DLP_AUDIO_FORMAT

logger = logging.getLogger("vid2txt")


class DownloadError(Exception):
    """Raised when video download fails."""


class ConversionError(Exception):
    """Raised when ffmpeg audio conversion fails."""


class Downloader:
    """Download audio from a video URL and convert to 16kHz mono WAV."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found. Install it: choco install ffmpeg"
            )

    def _run_ytdlp(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run yt-dlp with common flags."""
        cmd = ["yt-dlp"]
        if not self.verbose:
            cmd.append("--quiet")
            cmd.append("--no-warnings")
        cmd.extend(args)

        logger.debug("yt-dlp command: %s", " ".join(cmd))

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def get_video_info(self, url: str) -> dict:
        """Fetch video metadata without downloading (fast API call)."""
        logger.info("Fetching video info from: %s", url)

        result = self._run_ytdlp([
            "--dump-json",
            "--skip-download",
            url,
        ])

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "HTTP Error 404" in stderr:
                raise DownloadError("Video not found. Check the URL.")
            if "Private video" in stderr or "unavailable" in stderr.lower():
                raise DownloadError("Video is private or unavailable.")
            raise DownloadError(
                f"yt-dlp failed with exit code {result.returncode}:\n{stderr}"
            )

        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            raise DownloadError(f"Failed to parse video info: {e}")

    def download_audio(self, url: str, output_dir: str, video_id: str) -> str:
        """Download best audio stream from *url* into *output_dir*.

        Returns the path to the downloaded audio file.
        """
        output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

        logger.info("Downloading audio stream...")
        result = self._run_ytdlp([
            "--format", YT_DLP_AUDIO_FORMAT,
            "--output", output_template,
            url,
        ])

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise DownloadError(
                f"yt-dlp download failed with exit code {result.returncode}:\n{stderr}"
            )

        # Find the downloaded file in the output directory
        audio_files = [
            f for f in glob.glob(os.path.join(output_dir, "*"))
            if os.path.isfile(f) and not f.endswith(".wav")
        ]
        if not audio_files:
            # Retry: maybe yt-dlp used a different extension than expected
            audio_files = [
                f for f in glob.glob(os.path.join(output_dir, "*"))
                if os.path.isfile(f)
            ]

        if not audio_files:
            raise DownloadError(
                f"No audio file found in {output_dir} after download. "
                f"yt-dlp stderr: {result.stderr[:500]}"
            )

        audio_path = audio_files[0]
        logger.info("Downloaded: %s (%.1f MB)", os.path.basename(audio_path),
                     os.path.getsize(audio_path) / (1024 * 1024))
        return audio_path

    def convert_to_wav(self, input_path: str, output_path: str) -> str:
        """Convert *input_path* audio to 16kHz mono PCM WAV at *output_path*.

        Returns the output path.
        """
        cmd = [
            "ffmpeg",
            "-y",                     # overwrite output
            "-i", input_path,
            "-ar", str(SAMPLE_RATE),  # 16000 Hz
            "-ac", str(AUDIO_CHANNELS),  # mono
            "-c:a", AUDIO_CODEC,      # pcm_s16le
            "-loglevel", "error",     # only errors to stderr
            output_path,
        ]

        if self.verbose:
            cmd[cmd.index("-loglevel") + 1] = "info"

        logger.info("Converting audio to 16kHz mono WAV...")
        logger.debug("ffmpeg command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            raise ConversionError(
                f"ffmpeg conversion failed:\n{result.stderr.strip()}"
            )

        if not os.path.isfile(output_path):
            raise ConversionError(f"ffmpeg did not produce output file: {output_path}")

        logger.info("WAV ready: %s (%.1f MB)", os.path.basename(output_path),
                     os.path.getsize(output_path) / (1024 * 1024))
        return output_path

    def __call__(self, url: str) -> tuple[str, str, dict]:
        """Download audio and convert to WAV.

        Returns (wav_path, temp_dir, video_info).
        The caller is responsible for cleaning up temp_dir.
        """
        temp_dir = tempfile.mkdtemp(prefix="vid2txt_")
        try:
            # Step 1: Get video metadata (fast, no download)
            video_info = self.get_video_info(url)
            video_id = video_info.get("id", "")

            # Step 2: Download the audio
            audio_path = self.download_audio(url, temp_dir, video_id)

            # Step 3: Convert to WAV
            wav_path = os.path.join(temp_dir, "audio.wav")
            self.convert_to_wav(audio_path, wav_path)

        except Exception:
            # Clean up on failure — caller won't get temp_dir to clean
            import shutil as _shutil
            _shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return wav_path, temp_dir, video_info
