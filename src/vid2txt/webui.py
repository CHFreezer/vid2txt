"""WebUI for vid2txt — Gradio-based browser interface."""

# Must run before any CUDA-dependent imports
from src.vid2txt.cuda_setup import setup as _setup_cuda

_setup_cuda()

import os
import tempfile
import logging
from datetime import datetime

import gradio as gr

from src.vid2txt.config import DEFAULT_MODEL, SUPPORTED_MODELS
from src.vid2txt.downloader import Downloader, DownloadError, ConversionError
from src.vid2txt.transcriber import Transcriber, Segment
from src.vid2txt.formatter import Formatter
from src.vid2txt.utils import (
    validate_bilibili_url,
    get_output_basename,
    cleanup_temp_dir,
    format_timestamp,
    check_dependencies,
)

logger = logging.getLogger("vid2txt")

# Output directory for generated files
OUTPUT_DIR = os.path.join(os.getcwd(), "webui_outputs")

LANGUAGE_CHOICES = [
    ("自动检测", "auto"),
    ("中文 (zh)", "zh"),
    ("English (en)", "en"),
    ("日本語 (ja)", "ja"),
    ("한국어 (ko)", "ko"),
    ("Français (fr)", "fr"),
    ("Deutsch (de)", "de"),
    ("Español (es)", "es"),
    ("Русский (ru)", "ru"),
    ("ไทย (th)", "th"),
    ("Tiếng Việt (vi)", "vi"),
]

# ---------------------------------------------------------------------------
# Pipeline generator — the core logic
# ---------------------------------------------------------------------------


def _transcribe_pipeline(
    url: str,
    model_size: str,
    language: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Generator that runs the full pipeline with progressive UI updates.

    Yields tuples matching the output components:
    (status_md, preview_txt, summary_row, lang_md, duration_md,
     download_row, txt_file, srt_file)
    """
    _temp_dir: str | None = None
    all_segments: list[Segment] = []

    # Convenience: default tuple for "hidden everything except status"
    def _hidden(status: str) -> tuple:
        return (
            gr.update(value=status),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            None,
            None,
        )

    try:
        # ---- Phase 0: Validate URL ----
        if not validate_bilibili_url(url):
            yield _hidden("**❌ 无效的 Bilibili 链接，请检查 URL 格式。**")
            return

        yield _hidden("**① 已验证 URL**")
        progress(0.05, desc="准备下载...")

        # ---- Phase 1: Download ----
        yield _hidden("**② 正在获取视频信息...**")
        progress(0.08, desc="获取视频信息")

        downloader = Downloader(verbose=False)
        wav_path, _temp_dir, video_info = downloader(url)

        title = video_info.get("title", "")
        video_id = video_info.get("id", "")
        output_basename = get_output_basename(title, video_id)
        upload_date = video_info.get("upload_date", "")
        if len(upload_date) == 8:
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"

        progress(0.35, desc="已下载音频")

        yield (
            gr.update(value=f"**② 已下载音频** — {title}"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            None,
            None,
        )

        # ---- Phase 2: Transcribe (streaming) ----
        yield (
            gr.update(value="**③ 正在加载模型与转录...**（片段将实时显示在下方）"),
            gr.update(value="", visible=True),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            None,
            None,
        )
        progress(0.40, desc="转录中...")

        lang_param = None if language == "auto" else language
        transcriber = Transcriber(model_size=model_size)

        preview_lines: list[str] = []
        last_yield_count = -1

        for segment in transcriber.transcribe_stream(wav_path, language=lang_param):
            all_segments.append(segment)

            # Build SRT-like preview
            ts_start = format_timestamp(segment["start"])
            ts_end = format_timestamp(segment["end"])
            preview_lines.append(f"[{ts_start} → {ts_end}]  {segment['text']}")

            # Only yield UI updates every few segments to avoid thrashing
            seg_count = len(all_segments)
            if seg_count - last_yield_count >= 3 or seg_count <= 3:
                last_yield_count = seg_count
                preview_text = "\n".join(preview_lines)

                # Estimate progress from audio duration consumed
                audio_dur = getattr(transcriber, "_audio_duration", 1) or 1
                prog = min(0.88, 0.40 + 0.48 * (segment["end"] / max(audio_dur, 0.1)))

                progress(prog, desc=f"转录中...（{seg_count} 个片段）")

                yield (
                    gr.update(value=f"**③ 正在转录...** 已识别 **{seg_count}** 个片段"),
                    gr.update(value=preview_text),
                    gr.update(visible=False),
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                    None,
                    None,
                )

        # Final yield with all segments included
        preview_text = "\n".join(preview_lines)
        detected_lang = getattr(transcriber, "_detected_language", "?")
        detected_prob = getattr(transcriber, "_language_probability", 0) * 100
        audio_duration = getattr(transcriber, "_audio_duration", 0)

        progress(0.90, desc="转录完成")

        yield (
            gr.update(value=f"**③ 转录完成** — {len(all_segments)} 个片段"),
            gr.update(value=preview_text),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            None,
            None,
        )

        # ---- Phase 3: Write files ----
        yield (
            gr.update(value="**④ 正在写入输出文件...**"),
            gr.update(value=preview_text),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            None,
            None,
        )
        progress(0.95, desc="写入文件")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        formatter = Formatter()
        output_files = formatter.write(all_segments, OUTPUT_DIR, output_basename)

        # ---- Phase 4: Done ----
        progress(1.0, desc="完成!")

        word_count = sum(len(seg["text"]) for seg in all_segments)
        duration_str = f"{audio_duration:.0f}s" if audio_duration else "?"

        yield (
            gr.update(value="## ✅ 转录完成！"),
            gr.update(value=preview_text),
            gr.update(visible=True),
            gr.update(value=f"**语言：** {detected_lang}（{detected_prob:.1f}% 置信度）"),
            gr.update(
                value=f"**时长：** {duration_str}　|　**片段：** {len(all_segments)}　|　**字数：** {word_count}"
            ),
            gr.update(visible=True),
            output_files["txt"],
            output_files["srt"],
        )

    except DownloadError as e:
        yield _hidden(f"**❌ 下载失败：** {e}")
    except ConversionError as e:
        yield _hidden(f"**❌ 音频转换失败：** {e}")
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "oom" in msg.lower():
            yield _hidden("**❌ GPU 显存不足，请尝试更小的模型。**")
        else:
            yield _hidden(f"**❌ 模型错误：** {e}")
    except Exception as e:
        logger.exception("WebUI pipeline error")
        yield _hidden(f"**❌ 未知错误：** {e}")
    finally:
        # Always clean up temp files
        if _temp_dir:
            cleanup_temp_dir(_temp_dir)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


# Gradio custom CSS (Gradio 6+ expects this in launch(), not Blocks constructor)
_UI_CSS = """
footer { display: none !important; }
.preview-box textarea { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 13px; line-height: 1.6; }
"""


def _build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    with gr.Blocks(title="vid2txt — Bilibili 视频转文字") as demo:
        # ── Header ──
        gr.Markdown(
            """
            # 🎤 vid2txt
            **Bilibili 视频语音转文字** — 粘贴链接，自动下载音频并用 Whisper 转录。
            """
        )

        # ── Input area ──
        with gr.Row():
            url_input = gr.Textbox(
                label="🎬 Bilibili 视频链接",
                placeholder="https://www.bilibili.com/video/BV... 或 https://b23.tv/...",
                scale=5,
                container=True,
            )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(SUPPORTED_MODELS),
                value=DEFAULT_MODEL,
                label="📦 模型",
                scale=1,
            )
            language_dropdown = gr.Dropdown(
                choices=LANGUAGE_CHOICES,
                value="auto",
                label="🌐 语言",
                scale=1,
            )
            transcribe_btn = gr.Button("▶ 开始转录", variant="primary", scale=1, size="lg")

        # ── Status ──
        status_md = gr.Markdown(value="就绪，等待输入链接...")

        # ── Live preview ──
        preview_box = gr.Textbox(
            label="📝 实时转录预览",
            lines=18,
            max_lines=30,
            interactive=False,
            visible=False,
            elem_classes=["preview-box"],
        )

        # ── Summary row (hidden until done) ──
        with gr.Row(visible=False) as summary_row:
            lang_md = gr.Markdown()
            duration_md = gr.Markdown()

        # ── Download buttons (hidden until done) ──
        with gr.Row(visible=False) as download_row:
            txt_download = gr.File(label="📥 下载 TXT（纯文本）")
            srt_download = gr.File(label="📥 下载 SRT（字幕）")

        # ── Footer ──
        gr.Markdown(
            """
            ---
            💡 提示：模型越大准确率越高但速度越慢。| 首次使用会自动下载 Whisper 模型（约 1.5GB）。
            """
        )

        # ── Wire up ──
        transcribe_btn.click(
            fn=_transcribe_pipeline,
            inputs=[url_input, model_dropdown, language_dropdown],
            outputs=[
                status_md,
                preview_box,
                summary_row,
                lang_md,
                duration_md,
                download_row,
                txt_download,
                srt_download,
            ],
        )

    return demo


def main() -> None:
    """Launch the Gradio WebUI."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # Quick dependency check
    missing = check_dependencies()
    if missing:
        print("⚠️  缺少依赖：")
        for m in missing:
            print(f"  - {m}")
        print()
        print("WebUI 仍然可以启动，但转录时会失败。\n")

    demo = _build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True,
        css=_UI_CSS,
    )


if __name__ == "__main__":
    main()
