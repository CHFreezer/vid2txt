"""Output formatters — TXT plain text and SRT subtitle format."""

import os
import logging

from .transcriber import Segment
from .utils import format_timestamp
from .config import OUTPUT_ENCODING

logger = logging.getLogger("vid2txt")


def _has_translations(segments: list[Segment]) -> bool:
    """Return True if any segment has a translated_text."""
    return any(seg.get("translated_text") for seg in segments)


class Formatter:
    """Format transcription segments into TXT and SRT output."""

    # -- Plain text -------------------------------------------------------

    def to_txt(self, segments: list[Segment]) -> str:
        """Produce plain text from segments (original language)."""
        return "\n".join(seg["text"] for seg in segments)

    def to_translated_txt(self, segments: list[Segment]) -> str:
        """Produce plain text from translated segments."""
        return "\n".join(seg.get("translated_text") or "" for seg in segments)

    def to_bilingual_txt(self, segments: list[Segment]) -> str:
        """Produce bilingual text: original + translation alternating."""
        blocks: list[str] = []
        for seg in segments:
            ts_start = format_timestamp(seg["start"])
            ts_end = format_timestamp(seg["end"])
            blocks.append(
                f"[{ts_start} → {ts_end}]\n"
                f"🎙 {seg['text']}\n"
                f"🌐 {(seg.get('translated_text') or '')}\n"
            )
        return "\n".join(blocks)

    # -- SRT subtitle -----------------------------------------------------

    def to_srt(self, segments: list[Segment]) -> str:
        """Produce SRT subtitle content from segments (original language)."""
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp(seg["start"])
            end_ts = format_timestamp(seg["end"])
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg["text"])
            lines.append("")  # blank line between entries
        return "\n".join(lines)

    def to_translated_srt(self, segments: list[Segment]) -> str:
        """Produce SRT subtitle from translated text."""
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp(seg["start"])
            end_ts = format_timestamp(seg["end"])
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.get("translated_text", ""))
            lines.append("")
        return "\n".join(lines)

    def to_bilingual_srt(self, segments: list[Segment]) -> str:
        """Produce bilingual SRT: original + translation per entry."""
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp(seg["start"])
            end_ts = format_timestamp(seg["end"])
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"{seg['text']}\n{(seg.get('translated_text') or '')}")
            lines.append("")
        return "\n".join(lines)

    # -- Disk writer ------------------------------------------------------

    def write(
        self,
        segments: list[Segment],
        output_dir: str,
        basename: str,
        translated: bool = False,
        target_lang: str | None = None,
    ) -> dict[str, str]:
        """Write output files to *output_dir*.

        Always writes original-language TXT and SRT.  When *translated* is
        ``True``, also writes translated TXT/SRT and a bilingual TXT.

        Returns dict with paths keyed by format (txt, srt, translated_txt,
        translated_srt, bilingual_txt).
        """
        os.makedirs(output_dir, exist_ok=True)

        files: dict[str, str] = {}

        # Always write original-language output
        txt_path = os.path.join(output_dir, f"{basename}.txt")
        srt_path = os.path.join(output_dir, f"{basename}.srt")

        txt_content = self.to_txt(segments)
        srt_content = self.to_srt(segments)

        with open(txt_path, "w", encoding=OUTPUT_ENCODING) as f:
            f.write(txt_content)
        files["txt"] = txt_path
        logger.info("TXT written: %s (%d chars)", txt_path, len(txt_content))

        with open(srt_path, "w", encoding=OUTPUT_ENCODING) as f:
            f.write(srt_content)
        files["srt"] = srt_path
        logger.info("SRT written: %s (%d entries)", srt_path, len(segments))

        # Write translation outputs
        if translated and _has_translations(segments):
            lang_suffix = f".{target_lang}" if target_lang else ""

            # Translated TXT
            tr_txt_path = os.path.join(output_dir, f"{basename}{lang_suffix}.txt")
            tr_txt = self.to_translated_txt(segments)
            with open(tr_txt_path, "w", encoding=OUTPUT_ENCODING) as f:
                f.write(tr_txt)
            files["translated_txt"] = tr_txt_path
            logger.info("Translated TXT written: %s (%d chars)", tr_txt_path, len(tr_txt))

            # Translated SRT
            tr_srt_path = os.path.join(output_dir, f"{basename}{lang_suffix}.srt")
            tr_srt = self.to_translated_srt(segments)
            with open(tr_srt_path, "w", encoding=OUTPUT_ENCODING) as f:
                f.write(tr_srt)
            files["translated_srt"] = tr_srt_path
            logger.info("Translated SRT written: %s (%d entries)", tr_srt_path, len(segments))

            # Bilingual TXT
            bi_txt_path = os.path.join(output_dir, f"{basename}.bilingual.txt")
            bi_txt = self.to_bilingual_txt(segments)
            with open(bi_txt_path, "w", encoding=OUTPUT_ENCODING) as f:
                f.write(bi_txt)
            files["bilingual_txt"] = bi_txt_path
            logger.info("Bilingual TXT written: %s (%d chars)", bi_txt_path, len(bi_txt))

        return files
