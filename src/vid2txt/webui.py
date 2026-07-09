"""WebUI for vid2txt — Gradio-based browser interface.

Two-step workflow:
1.  **Analyse** — fetch video metadata, let user choose part/format
2.  **Transcribe** — download audio + transcribe with streaming preview

Device (CPU / CUDA) and model path are persisted in *vid2txt_config.json*.
"""

# Must run before any CUDA-dependent imports
from src.vid2txt.cuda_setup import setup as _setup_cuda

_setup_cuda()

import os
import logging
import time
from datetime import datetime
from typing import Generator

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
from src.vid2txt import model_manager
from src.vid2txt import settings

logger = logging.getLogger("vid2txt")

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

_UI_CSS = """
footer { display: none !important; }
.preview-box textarea { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 13px; line-height: 1.6; }
"""


# ======================================================================
# Pipeline (mostly unchanged from original)
# ======================================================================


def _transcribe_pipeline(
    url: str,
    model_size: str,
    language: str,
    device: str,
    model_path: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Generator that runs the full pipeline with progressive UI updates."""
    _temp_dir: str | None = None
    all_segments: list[Segment] = []

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
        compute_type = "float16" if device == "cuda" else "int8"
        transcriber = Transcriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            model_path=model_path if model_path else None,
        )

        preview_lines: list[str] = []
        last_yield_count = -1

        for segment in transcriber.transcribe_stream(wav_path, language=lang_param):
            all_segments.append(segment)

            ts_start = format_timestamp(segment["start"])
            ts_end = format_timestamp(segment["end"])
            preview_lines.append(f"[{ts_start} → {ts_end}]  {segment['text']}")

            seg_count = len(all_segments)
            if seg_count - last_yield_count >= 3 or seg_count <= 3:
                last_yield_count = seg_count
                preview_text = "\n".join(preview_lines)

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
        if _temp_dir:
            cleanup_temp_dir(_temp_dir)


# ======================================================================
# Analysis step (Step 1)
# ======================================================================


def _analyse_video(url: str) -> tuple:
    """Fetch video metadata and return UI updates for the analysis panel."""
    if not url or not validate_bilibili_url(url):
        return (
            gr.update(visible=False),  # analysis panel
            gr.update(visible=False),  # part selector
            gr.update(interactive=False),  # transcribe button
            "**❌ 无效的 Bilibili 链接**",
        )

    try:
        downloader = Downloader(verbose=False)
        parts = downloader.get_all_parts_info(url)
    except DownloadError as e:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=False),
            f"**❌ 获取视频信息失败：** {e}",
        )
    except Exception as e:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=False),
            f"**❌ 未知错误：** {e}",
        )

    if not parts:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=False),
            "**❌ 未找到视频信息**",
        )

    # Build analysis markdown from first part
    info = parts[0]
    title = info.get("title", "?")
    uploader = info.get("uploader", "?")
    duration = info.get("duration", 0) or 0
    mins, secs = divmod(int(duration), 60)
    duration_str = f"{mins}:{secs:02d}"
    view_count = info.get("view_count", 0) or 0
    upload_date = info.get("upload_date", "")
    if len(upload_date) == 8:
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"

    md = (
        f"### 📺 {title}\n\n"
        f"**UP主：** {uploader}　|　"
        f"**时长：** {duration_str}　|　"
        f"**播放：** {view_count:,}　|　"
        f"**日期：** {upload_date}"
    )

    # Multi-part selector
    if len(parts) > 1:
        part_choices = []
        for i, p in enumerate(parts):
            p_title = p.get("title", f"P{i + 1}")
            p_dur = p.get("duration", 0) or 0
            m, s = divmod(int(p_dur), 60)
            part_choices.append((f"{p_title}  ({m}:{s:02d})", i))
        part_dropdown = gr.update(choices=part_choices, value=0, visible=True)
    else:
        part_dropdown = gr.update(visible=False)

    return (
        gr.update(value=md, visible=True),
        part_dropdown,
        gr.update(interactive=True),
        f"✅ 分析完成 — {title}",
    )


# ======================================================================
# Model helpers
# ======================================================================


def _build_model_choices() -> list[tuple[str, str]]:
    """Build (label, value) tuples with download status."""
    model_path = settings.load().get("model_path", "./models")
    status = model_manager.list_models(model_path)
    choices = []
    for size in SUPPORTED_MODELS:
        s = status.get(size, {})
        if s.get("downloaded"):
            choices.append((f"[已下载] {size}", size))
        else:
            choices.append((f"[未下载] {size}", size))
    return choices


# (helper functions moved inside _build_ui to access UI component refs)


# ======================================================================
# Gradio UI
# ======================================================================


def _build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    user_settings = settings.load()
    initial_model_status = model_manager.list_models(user_settings.get("model_path", "./models"))
    default_downloaded = initial_model_status.get(DEFAULT_MODEL, {}).get("downloaded", True)

    with gr.Blocks(title="vid2txt — Bilibili 视频转文字") as demo:
        # ── Header ──
        gr.Markdown(
            """
            # 🎤 vid2txt
            **Bilibili 视频语音转文字** — 粘贴链接，先分析再转录。
            """
        )

        # ═══════════════════════════════════════════════════════════
        # Step 1: URL + Analyse
        # ═══════════════════════════════════════════════════════════
        with gr.Row(equal_height=True):
            url_input = gr.Textbox(
                label="🎬 Bilibili 视频链接",
                placeholder="https://www.bilibili.com/video/BV...",
                scale=5,
            )
            analyse_btn = gr.Button("🔍 分析", variant="secondary", scale=1)

        # Analysis results
        analysis_md = gr.Markdown(visible=False)
        with gr.Row(visible=False) as part_row:
            part_selector = gr.Dropdown(
                label="🎵 选择分P",
                choices=[],
                interactive=True,
            )

        # ═══════════════════════════════════════════════════════════
        # Settings
        # ═══════════════════════════════════════════════════════════
        with gr.Row(equal_height=True):
            model_path_box = gr.Textbox(
                label="💾 模型路径",
                value=user_settings.get("model_path", "./models"),
                scale=2,
            )
            model_dropdown = gr.Dropdown(
                choices=_build_model_choices(),
                value=DEFAULT_MODEL,
                label="📦 模型",
                scale=2,
                interactive=True,
            )
            download_model_btn = gr.Button(
                "📥 下载模型",
                variant="secondary",
                scale=1,
                visible=not default_downloaded,
            )

        with gr.Row():
            language_dropdown = gr.Dropdown(
                choices=LANGUAGE_CHOICES,
                value="auto",
                label="🌐 语言",
                scale=1,
            )
            device_radio = gr.Radio(
                choices=[("CPU", "cpu"), ("CUDA", "cuda")],
                value=user_settings.get("device", "cpu"),
                label="🖥 设备",
                scale=1,
            )

        with gr.Row():
            transcribe_btn = gr.Button(
                "▶ 开始转录",
                variant="primary",
                scale=1,
                size="lg",
                interactive=False,
            )

        # ═══════════════════════════════════════════════════════════
        # Status + Preview + Download
        # ═══════════════════════════════════════════════════════════
        status_md = gr.Markdown(value="就绪 — 请粘贴链接后点击 **分析**。")

        preview_box = gr.Textbox(
            label="📝 实时转录预览",
            lines=18,
            max_lines=30,
            interactive=False,
            visible=False,
            elem_classes=["preview-box"],
        )

        with gr.Row(visible=False) as summary_row:
            lang_md = gr.Markdown()
            duration_md = gr.Markdown()

        with gr.Row(visible=False) as download_row:
            txt_download = gr.File(label="📥 下载 TXT（纯文本）")
            srt_download = gr.File(label="📥 下载 SRT（字幕）")

        # ── Footer ──
        gr.Markdown(
            """
            ---
            💡 提示：首次使用需下载模型（约 150MB~3.5GB）。| 模型越大准确率越高。
            """
        )

        # ═══════════════════════════════════════════════════════════
        # Event handlers (defined here to access UI component refs)
        # ═══════════════════════════════════════════════════════════

        def on_model_select(model_size: str):
            path = model_path_box.value or "./models"
            status = model_manager.list_models(path)
            s = status.get(model_size, {})
            return gr.update(visible=not s.get("downloaded"))

        download_state = gr.State(value=(None, None))  # (Event | None, Thread | None)

        def on_download_model(model_size: str, path: str, state, progress=gr.Progress()):
            evt, _thread = state or (None, None)
            if evt is not None:
                evt.set()
                return (
                    gr.update(value="📥 下载模型", variant="secondary", visible=True),
                    gr.update(choices=_build_model_choices()),
                    (None, None),
                    "**⏹ 下载已取消**",
                )
            if not model_size:
                return (
                    gr.update(visible=False),
                    gr.update(choices=_build_model_choices()),
                    (None, None),
                    "**❌ 未选择模型**",
                )

            import threading
            cancel_evt = threading.Event()

            def _download_thread():
                def _cb(ratio: float):
                    progress(ratio, desc=f"下载 {model_size}...")
                try:
                    model_manager.download_model(
                        model_size, path or "./models",
                        progress_callback=_cb, cancel_event=cancel_evt,
                    )
                except RuntimeError:
                    pass

            t = threading.Thread(target=_download_thread, daemon=True)
            t.start()

            return (
                gr.update(value="⏹ 取消下载", variant="stop", visible=True),
                gr.update(),
                (cancel_evt, t),
                f"⏳ 正在下载 {model_size}...",
            )

        def on_poll_download(state):
            evt, thread = state or (None, None)
            if evt is None:
                return gr.update(), gr.update(), (None, None), gr.update()
            if not thread.is_alive():
                return (
                    gr.update(value="📥 下载模型", variant="secondary", visible=True),
                    gr.update(choices=_build_model_choices()),
                    (None, None),
                    "✅ 下载完成！",
                )
            if evt.is_set():
                return (
                    gr.update(value="📥 下载模型", variant="secondary", visible=True),
                    gr.update(choices=_build_model_choices()),
                    (None, None),
                    "**⏹ 下载已取消**",
                )
            return gr.update(), gr.update(), state, gr.update()

        def on_save_device(device: str):
            settings.save(device=device)

        def on_save_model_path(path: str):
            settings.save(model_path=path)
            return gr.update(choices=_build_model_choices())

        # ═══════════════════════════════════════════════════════════
        # Wire up events
        # ═══════════════════════════════════════════════════════════

        # Analyse button
        analyse_btn.click(
            fn=_analyse_video,
            inputs=[url_input],
            outputs=[analysis_md, part_row, transcribe_btn, status_md],
        )

        # Model dropdown → show/hide download button
        model_dropdown.change(
            fn=on_model_select,
            inputs=[model_dropdown],
            outputs=[download_model_btn],
        )

        # Download model button (toggle between download and cancel)
        download_model_btn.click(
            fn=on_download_model,
            inputs=[model_dropdown, model_path_box, download_state],
            outputs=[download_model_btn, model_dropdown, download_state, status_md],
        )

        # Timer to poll download completion
        gr.Timer(value=1.0).tick(
            fn=on_poll_download,
            inputs=[download_state],
            outputs=[download_model_btn, model_dropdown, download_state, status_md],
        )

        # Persist settings on change
        device_radio.change(fn=on_save_device, inputs=[device_radio], outputs=[])
        model_path_box.change(
            fn=on_save_model_path,
            inputs=[model_path_box],
            outputs=[model_dropdown],
        )

        # Transcribe button
        transcribe_btn.click(
            fn=_transcribe_pipeline,
            inputs=[
                url_input,
                model_dropdown,
                language_dropdown,
                device_radio,
                model_path_box,
            ],
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


# ======================================================================
# Launch
# ======================================================================


def main() -> None:
    """Launch the Gradio WebUI."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    missing = check_dependencies()
    if missing:
        print("⚠️  缺少依赖：")
        for m in missing:
            print(f"  - {m}")
        print()

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
