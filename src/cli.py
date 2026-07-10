"""CLI entry point for vid2txt."""

import argparse
import sys
import os
import logging

# --- Fix CUDA DLL loading on Windows (must run before any CUDA imports) ---
from src.cuda_setup import setup as _setup_cuda
_setup_cuda()

from src.config import DEFAULT_MODEL, SUPPORTED_MODELS, DEFAULT_TARGET_LANG
from src import __version__, settings
from src.utils import (
    validate_url,
    get_output_basename,
    cleanup_temp_dir,
    check_dependencies,
)
from src.downloader import Downloader, DownloadError, ConversionError
from src.transcriber import Transcriber
from src.translator import Translator, TranslationModelNotFoundError
from src.formatter import Formatter
from src.translation_model_manager import is_model_downloaded

logger = logging.getLogger("vid2txt")

# Exit codes
EXIT_OK = 0
EXIT_INVALID_URL = 1
EXIT_DOWNLOAD = 2
EXIT_NO_FFMPEG = 3
EXIT_CONVERSION = 4
EXIT_MODEL = 5
EXIT_IO = 6
EXIT_GPU_OOM = 7
EXIT_UNKNOWN = 8
EXIT_TRANSLATION_MODEL = 9
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vid2txt",
        description="Extract spoken text from Bilibili videos using Whisper speech recognition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python main.py https://www.bilibili.com/video/BV1GJ41177UQ\n"
               "  python main.py https://www.bilibili.com/video/BV1GJ41177UQ -o ./transcripts -m large-v3\n"
               "  python main.py https://b23.tv/xxxxxx --language zh -v\n"
               "  python -m src https://www.bilibili.com/video/BV1GJ41177UQ",
    )
    parser.add_argument(
        "url",
        help="Video URL (bilibili.com or b23.tv)",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config file (default: <project_root>/vid2txt_config.json).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory for TXT/SRT files (default: ./output)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help=f"Whisper model size (default: {DEFAULT_MODEL}). "
             "Larger = more accurate but slower / more memory.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for transcription (e.g. zh, en, ja). "
             "Auto-detected from audio if not specified.",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep temporary audio files after processing (for debugging).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"vid2txt {__version__}",
        help="Show version number and exit.",
    )
    # Translation
    parser.add_argument(
        "--translate",
        action="store_true",
        default=False,
        help="Enable translation after transcription (M2M100 via CTranslate2).",
    )
    parser.add_argument(
        "-t", "--target-lang",
        default=DEFAULT_TARGET_LANG,
        help=f"Target language code for translation (default: {DEFAULT_TARGET_LANG}).",
    )
    parser.add_argument(
        "--translation-model-path",
        default=None,
        help="Directory for translation model.  Uses config value if not set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    # Load config
    if args.config:
        settings.set_config_path(args.config)
    cfg = settings.load()

    logger.info("vid2txt %s", __version__)

    # --- Pre-flight checks ---

    # 1. Validate URL
    if not validate_url(args.url):
        logger.error("Invalid Bilibili URL: %s", args.url)
        logger.error("Expected format: https://www.bilibili.com/video/BV... or https://b23.tv/...")
        return EXIT_INVALID_URL

    # 2. Check external dependencies
    missing = check_dependencies()
    if missing:
        for m in missing:
            logger.error("Missing dependency: %s", m)
        return EXIT_NO_FFMPEG

    # --- Pipeline ---

    temp_dir: str | None = None

    try:
        # Phase 1: Download audio
        downloader = Downloader(verbose=args.verbose)
        wav_path, temp_dir, video_info = downloader(args.url)

        # Phase 2: Transcribe
        transcriber = Transcriber(
            model_size=args.model,
            whisper_model_path=cfg["whisper_model_path"],
        )
        result = transcriber.transcribe(wav_path, language=args.language)

        if not result["segments"]:
            logger.warning("No speech detected in the audio.")

        # Phase 3: Translate (optional)
        translated = False
        if args.translate:
            tl_path = args.translation_model_path or cfg["translation_model_path"]
            if not is_model_downloaded(tl_path):
                logger.error("Translation model not found at %s.", tl_path)
                logger.error(
                    "Download it first: python -c \"from src.translation_model_manager "
                    "import download_translation_model; download_translation_model('%s')\"",
                    tl_path,
                )
                return EXIT_TRANSLATION_MODEL

            # Resolve device for translation
            try:
                from ctranslate2 import get_cuda_device_count
                tl_device = "cuda" if get_cuda_device_count() > 0 else "cpu"
            except Exception:
                tl_device = "cpu"

            translator = Translator(
                model_path=tl_path,
                device=tl_device,
            )

            result["segments"] = translator.translate_segments(
                result["segments"],
                source_lang=result.get("language", "auto"),
                target_lang=args.target_lang,
            )
            translated = True
            logger.info("Translation complete: %s → %s",
                        result.get("language", "auto"), args.target_lang)

        # Phase 4: Format output
        title = video_info.get("title", "")
        video_id = video_info.get("id", "")
        basename = get_output_basename(title, video_id)

        formatter = Formatter()
        output_files = formatter.write(
            result["segments"],
            args.output_dir,
            basename,
            translated=translated,
            target_lang=args.target_lang if translated else None,
        )

        # --- Summary ---
        char_count = sum(len(seg["text"]) for seg in result["segments"])
        lang = result.get("language", "?")
        lang_prob = result.get("language_probability", 0) * 100
        duration = result.get("duration", 0)

        print()  # blank line before summary
        logger.info("=" * 50)
        logger.info("Done!")
        logger.info("  TXT: %s", output_files.get("txt", ""))
        logger.info("  SRT: %s", output_files.get("srt", ""))
        if translated:
            logger.info("  Translated TXT: %s", output_files.get("translated_txt", ""))
            logger.info("  Translated SRT: %s", output_files.get("translated_srt", ""))
            logger.info("  Bilingual TXT: %s", output_files.get("bilingual_txt", ""))
        logger.info("  Language: %s (%.1f%% confidence)", lang, lang_prob)
        logger.info("  Duration: %.1f seconds", duration)
        logger.info("  Characters: %d", char_count)
        logger.info("  Segments: %d", len(result["segments"]))
        logger.info("=" * 50)

        return EXIT_OK

    except DownloadError as e:
        logger.error("Download failed: %s", e)
        return EXIT_DOWNLOAD

    except TranslationModelNotFoundError as e:
        logger.error("Translation model error: %s", e)
        return EXIT_TRANSLATION_MODEL

    except ConversionError as e:
        logger.error("Audio conversion failed: %s", e)
        return EXIT_CONVERSION

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "OOM" in msg:
            logger.error("GPU out of memory. Try a smaller model (--model small) or CPU mode.")
            return EXIT_GPU_OOM
        logger.error("Model error: %s", e)
        return EXIT_MODEL

    except OSError as e:
        logger.error("I/O error: %s", e)
        return EXIT_IO

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return EXIT_INTERRUPTED

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_UNKNOWN

    finally:
        # Always clean up temp files unless user wants to keep them
        if temp_dir and not args.keep_files:
            cleanup_temp_dir(temp_dir)
        elif temp_dir and args.keep_files:
            logger.info("Temp files kept at: %s", temp_dir)
