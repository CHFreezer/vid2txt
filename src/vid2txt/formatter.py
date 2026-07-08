"""Output formatters — TXT plain text and SRT subtitle format."""

import os
import logging

from .transcriber import Segment
from .utils import format_timestamp
from .config import OUTPUT_ENCODING

logger = logging.getLogger("vid2txt")


class Formatter:
    """Format transcription segments into TXT and SRT output."""

    def to_txt(self, segments: list[Segment]) -> str:
        """Produce plain text from segments."""
        return "\n".join(seg["text"] for seg in segments)

    def to_srt(self, segments: list[Segment]) -> str:
        """Produce SRT subtitle content from segments."""
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp(seg["start"])
            end_ts = format_timestamp(seg["end"])
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg["text"])
            lines.append("")  # blank line between entries
        return "\n".join(lines)

    def write(
        self,
        segments: list[Segment],
        output_dir: str,
        basename: str,
    ) -> dict[str, str]:
        """Write TXT and SRT files to *output_dir*.

        Returns dict with 'txt' and 'srt' paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        txt_path = os.path.join(output_dir, f"{basename}.txt")
        srt_path = os.path.join(output_dir, f"{basename}.srt")

        txt_content = self.to_txt(segments)
        srt_content = self.to_srt(segments)

        with open(txt_path, "w", encoding=OUTPUT_ENCODING) as f:
            f.write(txt_content)

        with open(srt_path, "w", encoding=OUTPUT_ENCODING) as f:
            f.write(srt_content)

        logger.info("TXT written: %s (%d chars)", txt_path, len(txt_content))
        logger.info("SRT written: %s (%d entries)", srt_path, len(segments))

        return {"txt": txt_path, "srt": srt_path}
