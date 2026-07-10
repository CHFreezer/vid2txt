"""WebUI for vid2txt — Gradio-based browser interface.

Two-step workflow:
1.  **Analyse** — fetch video metadata, let user choose part/format
2.  **Transcribe** — download audio + transcribe with streaming preview

Device (CPU / CUDA) and model path are persisted in *vid2txt_config.json*.
"""

# Must run before any CUDA-dependent imports
from src.cuda_setup import setup as _setup_cuda

_setup_cuda()

import os
import logging

import gradio as gr

from src.config import (
    DEFAULT_MODEL, SUPPORTED_MODELS,
    TARGET_LANGUAGE_CHOICES,
    DEFAULT_WHISPER_MODEL_DIR, DEFAULT_TRANSLATION_MODEL_DIR,
)
from src.downloader import Downloader, DownloadError, ConversionError
from src.transcriber import Transcriber, Segment, ModelNotFoundError
from src.translator import Translator, TranslationModelNotFoundError
from src.formatter import Formatter
from src.utils import (
    validate_url,
    get_output_basename,
    cleanup_temp_dir,
    format_timestamp,
    check_dependencies,
)
from src import model_manager
from src import translation_model_manager
from src import settings

logger = logging.getLogger("vid2txt")

OUTPUT_DIR = os.path.join(os.getcwd(), "webui_outputs")

LANGUAGE_CHOICES = [
    ("自动检测", "auto"),
    ("中文", "zh"),
    ("English", "en"),
    ("日本語", "ja"),
    ("한국어", "ko"),
]

_UI_CSS = """
footer { display: none !important; }
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
.preview-box textarea { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 13px; line-height: 1.6; }
"""


# ======================================================================
# Pipeline
# ======================================================================

import threading
_current_stop_event: list[threading.Event] = []  # mutable cell for per-pipeline Event

# Pre-built "hidden" UI tuple used across yield points to hide intermediate outputs.
# Positions: status_md, preview_box, summary_row, lang_md, duration_md,
#             download_row, txt_download, srt_download, stop_btn, transcribe_btn
_HIDDEN_OUTPUTS = (
    gr.update(),                                      # 0: status_md  (set per call)
    gr.update(visible=False),                         # 1: preview_box
    gr.update(visible=False),                         # 2: summary_row
    gr.update(),                                      # 3: lang_md
    gr.update(),                                      # 4: duration_md
    gr.update(visible=False),                         # 5: download_row
    None,                                             # 6: txt_download
    None,                                             # 7: srt_download
    gr.update(visible=False),                         # 8: stop_btn
    gr.update(value="▶ 开始转录", variant="primary", interactive=True, visible=True),  # 9: transcribe_btn
)

# Like _HIDDEN_OUTPUTS but keeps the preview_box visible — used on stop so
# already-transcribed text is not lost.
_STOP_OUTPUTS = (
    gr.update(),                                      # 0: status_md
    gr.update(),                                      # 1: preview_box — KEEP VISIBLE
    gr.update(visible=False),                         # 2: summary_row
    gr.update(),                                      # 3: lang_md
    gr.update(),                                      # 4: duration_md
    gr.update(visible=False),                         # 5: download_row
    None,                                             # 6: txt_download
    None,                                             # 7: srt_download
    gr.update(visible=False),                         # 8: stop_btn
    gr.update(value="▶ 开始转录", variant="primary", interactive=True, visible=True),  # 9: transcribe_btn
)


def _transcribe_pipeline(
    url: str,
    model_size: str,
    language: str,
    device: str,
    whisper_model_path: str,
    translate_enabled: bool,
    target_lang: str,
    translation_model_path: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Generator that runs the full pipeline with progressive UI updates."""
    stop_event = threading.Event()
    _current_stop_event.clear()
    _current_stop_event.append(stop_event)
    _temp_dir: str | None = None
    all_segments: list[Segment] = []

    def _hidden(status: str) -> tuple:
        return (
            gr.update(value=status),
            *_HIDDEN_OUTPUTS[1:],
        )

    def _stopped(status: str) -> tuple:
        """Like _hidden but keeps preview_box visible so transcribed text
        is preserved after the user clicks stop."""
        return (
            gr.update(value=status),
            *_STOP_OUTPUTS[1:],
        )

    try:
        # ---- Phase 0: Validate URL ----
        if not validate_url(url):
            yield _hidden("**❌ 无效的视频链接，请检查 URL 格式。**")
            return

        yield _hidden("**① 已验证 URL**")
        progress(0.05, desc="准备下载...")

        # ---- Phase 1: Download ----
        yield _hidden("**② 正在获取视频信息...**")
        progress(0.08, desc="获取视频信息")

        downloader = Downloader(verbose=False)
        wav_path, _temp_dir, video_info = downloader(url)

        # Stop may have been requested during the blocking download —
        # check before proceeding to transcription.
        if stop_event.is_set():
            yield _stopped("**⏹ 转录已停止**")
            return

        title = video_info.get("title", "")
        video_id = video_info.get("id", "")
        output_basename = get_output_basename(title, video_id)
        upload_date = video_info.get("upload_date", "")
        if len(upload_date) == 8:
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"

        progress(0.35, desc="已下载音频")

        yield _hidden(f"**② 已下载音频** — {title}")

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
            gr.update(visible=True),
            gr.update(visible=False),
        )
        progress(0.40, desc="转录中...")

        lang_param = None if language == "auto" else language
        compute_type = "float16" if device == "cuda" else "int8"
        transcriber = Transcriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            whisper_model_path=whisper_model_path or DEFAULT_WHISPER_MODEL_DIR,
        )

        preview_lines: list[str] = []
        last_yield_count = -1

        # Pre-load the model so stop_event can interrupt before the
        # potentially long _load_model() call inside transcribe_stream.
        try:
            transcriber._load_model()
        except ModelNotFoundError as e:
            yield _hidden(f"**❌ 模型未找到：** {e}")
            return

        if stop_event.is_set():
            yield _stopped("**⏹ 转录已停止**")
            return

        for segment in transcriber.transcribe_stream(wav_path, language=lang_param):
            if stop_event.is_set():
                break
            all_segments.append(segment)

            ts_start = format_timestamp(segment["start"])
            ts_end = format_timestamp(segment["end"])
            preview_lines.append(f"[{ts_start} → {ts_end}]  {segment['text']}")

            seg_count = len(all_segments)
            if seg_count - last_yield_count >= 3 or seg_count <= 3:
                last_yield_count = seg_count
                preview_text = "\n".join(preview_lines)

                audio_dur = (transcriber.info.duration if transcriber.info else 1) or 1
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
                    gr.update(visible=True),
                    gr.update(visible=False),
                )

        if stop_event.is_set():
            yield _stopped("**⏹ 转录已停止**")
            return

        preview_text = "\n".join(preview_lines)
        stream_info = transcriber.info
        detected_lang = stream_info.language if stream_info else "?"
        detected_prob = (stream_info.language_probability if stream_info else 0) * 100
        audio_duration = stream_info.duration if stream_info else 0

        progress(0.65, desc="转录完成")

        # ---- Phase 3: Translate (optional) ----
        translated = False
        if translate_enabled and all_segments:
            source_lang = stream_info.language if stream_info else "auto"
            source_lang = source_lang if source_lang else "auto"

            # Check model
            if not translation_model_manager.is_model_downloaded(
                translation_model_path
            ):
                yield _hidden(
                    f"**❌ 翻译模型未下载**\n\n"
                    f"请先在设置中下载翻译模型。"
                )
                return

            tl_device = device
            tl_compute = "float16" if device == "cuda" else "int8"

            yield (
                gr.update(value="**③ 正在翻译...**"),
                gr.update(value=preview_text),
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(visible=False),
                None,
                None,
                gr.update(visible=True),
                gr.update(visible=False),
            )
            progress(0.68, desc="加载翻译模型...")

            translator = Translator(
                model_path=translation_model_path,
                device=tl_device,
                compute_type=tl_compute,
            )

            translated_preview_lines: list[str] = []
            translated_segments: list[Segment] = []
            last_tl_yield = -1
            total = len(all_segments)

            for i, seg in enumerate(translator.translate_segments_stream(
                all_segments, source_lang=source_lang, target_lang=target_lang
            )):
                if stop_event.is_set():
                    break
                translated_segments.append(seg)

                ts_s = format_timestamp(seg["start"])
                ts_e = format_timestamp(seg["end"])
                translated_preview_lines.append(
                    f"[{ts_s} → {ts_e}]\n"
                    f"🎙 {seg['text']}\n"
                    f"🌐 {seg.get('translated_text', '')}"
                )

                count = len(translated_segments)
                if count - last_tl_yield >= 2 or count <= 2 or count == total:
                    last_tl_yield = count
                    tl_preview = "\n\n".join(translated_preview_lines)
                    prog = 0.68 + 0.22 * (count / total)
                    progress(prog, desc=f"翻译中...（{count}/{total}）")

                    yield (
                        gr.update(value=f"**③ 正在翻译...** {count}/{total} 个片段"),
                        gr.update(value=tl_preview),
                        gr.update(visible=False),
                        gr.update(),
                        gr.update(),
                        gr.update(visible=False),
                        None,
                        None,
                        gr.update(visible=True),
                        gr.update(visible=False),
                    )

            if stop_event.is_set():
                yield _stopped("**⏹ 转录已停止**")
                return

            all_segments = translated_segments
            preview_text = "\n\n".join(translated_preview_lines)
            translated = True
            progress(0.90, desc="翻译完成")

        # ---- Phase 4: Write files ----
        yield (
            gr.update(value="**④ 正在写入输出文件...**"),
            gr.update(value=preview_text),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            None,
            None,
            gr.update(visible=False),
            gr.update(value="▶ 开始转录", variant="primary", interactive=True, visible=True),
        )
        progress(0.95, desc="写入文件")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        formatter = Formatter()
        output_files = formatter.write(
            all_segments, OUTPUT_DIR, output_basename,
            translated=translated, target_lang=target_lang if translated else None,
        )

        # ---- Done ----
        progress(1.0, desc="完成!")

        word_count = sum(len(seg["text"]) for seg in all_segments)
        duration_str = f"{audio_duration:.0f}s" if audio_duration else "?"

        result_status = "## ✅ 转录+翻译完成！" if translated else "## ✅ 转录完成！"
        yield (
            gr.update(value=result_status),
            gr.update(value=preview_text),
            gr.update(visible=True),
            gr.update(value=f"**语言：** {detected_lang}（{detected_prob:.1f}% 置信度）"),
            gr.update(
                value=f"**时长：** {duration_str}　|　**片段：** {len(all_segments)}　|　**字数：** {word_count}"
            ),
            gr.update(visible=True),
            output_files["txt"],
            output_files["srt"],
            gr.update(visible=False),
            gr.update(value="▶ 开始转录", variant="primary", interactive=True, visible=True),
        )

    except DownloadError as e:
        yield _hidden(f"**❌ 下载失败：** {e}")
    except TranslationModelNotFoundError as e:
        yield _hidden(f"**❌ 翻译模型错误：** {e}")
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


def _list_audio_formats(part: dict) -> list[dict]:
    """Return audio-only formats sorted by bitrate descending."""
    formats = part.get("formats", [])
    audio = []
    for fmt in formats:
        acodec = fmt.get("acodec", "none")
        vcodec = fmt.get("vcodec", "none")
        abr = fmt.get("abr") or 0
        if acodec == "none" or vcodec != "none" or abr <= 0:
            continue
        note = fmt.get("format_note", "")
        if "Dolby" in note or "Hi-Res" in note or "会员" in note:
            continue
        audio.append(fmt)
    audio.sort(key=lambda f: f.get("abr") or 0, reverse=True)
    return audio


def _analyse_video(url: str) -> tuple:
    """Fetch video metadata and return UI updates for the analysis panel."""
    if not url or not validate_url(url):
        return (
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(interactive=False),
            "**❌ 无效的视频链接**",
        )

    try:
        downloader = Downloader(verbose=False)
        parts = downloader.get_all_parts_info(url)
    except DownloadError as e:
        return (
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(interactive=False),
            f"**❌ 获取视频信息失败：** {e}",
        )
    except Exception:
        logger.exception("Unexpected error during video analysis")
        return (
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(interactive=False),
            "**❌ 未知错误，请查看控制台日志。**",
        )

    if not parts:
        return (
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(interactive=False),
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

    audio_formats = _list_audio_formats(info)
    audio_count = len(audio_formats)

    md = (
        f"### 📺 {title}\n\n"
        f"**UP主：** {uploader}　|　"
        f"**时长：** {duration_str}　|　"
        f"**播放：** {view_count:,}　|　"
        f"**日期：** {upload_date}\n\n"
        f"**分P：** {len(parts)}　|　**音频流：** {audio_count}"
    )

    # Audio format selector — sorted by quality, default to best
    fmt_choices = []
    for fmt in audio_formats:
        codec = fmt.get("acodec", "?")
        abr = fmt.get("abr") or 0
        size = fmt.get("filesize") or fmt.get("filesize_approx") or 0
        size_str = f"~{size / 1024 / 1024:.0f} MB" if size else "?"
        fmt_choices.append((f"{codec} @ {abr:.0f}kbps ({size_str})", fmt.get("format_id", "")))

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

    fmt_dropdown = gr.update(
        choices=fmt_choices,
        value=fmt_choices[0][1] if fmt_choices else None,
        visible=len(fmt_choices) > 0,
    )

    return (
        gr.update(value=md, visible=True),
        part_dropdown,
        fmt_dropdown,
        gr.update(interactive=True),
        f"✅ 分析完成 — {title}",
    )


# ======================================================================
# Model helpers
# ======================================================================


def _build_model_choices() -> list[tuple[str, str]]:
    """Build (label, value) tuples with download status."""
    whisper_model_path = settings.load().get("whisper_model_path", DEFAULT_WHISPER_MODEL_DIR)
    status = model_manager.list_models(whisper_model_path)
    choices = []
    for size in SUPPORTED_MODELS:
        s = status.get(size, {})
        if s.get("downloaded"):
            choices.append((f"[已下载] {size}", size))
        else:
            choices.append((f"[未下载] {size}", size))
    return choices


def _refresh_model_list(path: str) -> tuple:
    """Re-scan model directory and update dropdown + download button."""
    path = path or DEFAULT_WHISPER_MODEL_DIR
    status = model_manager.list_models(path)
    new_choices = []
    for size in SUPPORTED_MODELS:
        s = status.get(size, {})
        if s.get("downloaded"):
            new_choices.append((f"[已下载] {size}", size))
        else:
            new_choices.append((f"[未下载] {size}", size))
    model = settings.load().get("model", DEFAULT_MODEL)
    s = status.get(model, {})
    return (
        gr.update(choices=new_choices),
        gr.update(visible=not s.get("downloaded")),
    )


# (helper functions moved inside _build_ui to access UI component refs)


def _should_show_translation_download(s: dict) -> bool:
    """Return True if the download button should be visible."""
    if not s.get("translate_enabled"):
        return False
    model_path = s.get("translation_model_path", DEFAULT_TRANSLATION_MODEL_DIR)
    return not translation_model_manager.is_model_downloaded(model_path)


def _translation_model_info(s: dict) -> str:
    """Return a Markdown line describing the translation model status."""
    model_path = s.get("translation_model_path", DEFAULT_TRANSLATION_MODEL_DIR)
    if translation_model_manager.is_model_downloaded(model_path):
        return "✅ 翻译模型已就绪（M2M100-418M int8，~500MB）"
    return "💡 M2M100-418M int8，约500MB，100种语言。点击右侧按钮下载。"


# ======================================================================
# Gradio UI
# ======================================================================


def _build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    user_settings = settings.load()
    initial_model_status = model_manager.list_models(user_settings.get("whisper_model_path", DEFAULT_WHISPER_MODEL_DIR))
    default_downloaded = initial_model_status.get(DEFAULT_MODEL, {}).get("downloaded", True)

    with gr.Blocks(title="vid2txt — 视频转文字（Bilibili / YouTube / Shorts）") as demo:
        # ── Header ──
        gr.Markdown(
            """
            # 🎤 vid2txt
            **视频语音转文字** — 粘贴链接，先分析再转录
            """
        )

        # ═══════════════════════════════════════════════════════════
        # Step 1: URL + Analyse
        # ═══════════════════════════════════════════════════════════
        with gr.Row(equal_height=True):
            url_input = gr.Textbox(
                label="视频地址",
                placeholder="粘贴视频链接...（Bilibili / YouTube / Shorts）",
                scale=6,
            )
            analyse_btn = gr.Button("🔍 分析", variant="secondary", scale=1)

        # Analysis results
        analysis_md = gr.Markdown(visible=False)
        with gr.Row() as part_row:
            part_selector = gr.Dropdown(
                label="选择分P",
                choices=[],
                interactive=True,
                visible=False,
                scale=1,
            )
            audio_selector = gr.Dropdown(
                label="音频流",
                choices=[],
                interactive=True,
                visible=False,
                scale=2,
            )

        # ═══════════════════════════════════════════════════════════
        # Settings (collapsible)
        # ═══════════════════════════════════════════════════════════
        with gr.Accordion("⚙ 模型与设备设置", open=True):
            with gr.Row(equal_height=True):
                whisper_model_path_box = gr.Textbox(
                    label="模型存储路径",
                    value=user_settings.get("whisper_model_path", DEFAULT_WHISPER_MODEL_DIR),
                    scale=2,
                )
                model_dropdown = gr.Dropdown(
                    choices=_build_model_choices(),
                    value=user_settings.get("model", DEFAULT_MODEL),
                    label="Whisper 模型",
                    scale=2,
                    interactive=True,
                )
                refresh_models_btn = gr.Button(
                    "🔄",
                    variant="secondary",
                    scale=0,
                    min_width=40,
                )
                download_model_btn = gr.Button(
                    "⬇ 下载模型",
                    variant="secondary",
                    scale=1,
                    visible=not default_downloaded,
                )

            gr.Markdown("💡 模型越大准确率越高，但速度越慢。首次使用需下载模型（150MB~3.5GB）。")

            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value=user_settings.get("language", "auto"),
                    label="语言",
                    scale=1,
                )
                device_radio = gr.Radio(
                    choices=[("💻 CPU", "cpu"), ("⚡ CUDA", "cuda")],
                    value=user_settings.get("device", "cpu"),
                    label="推理设备",
                    scale=1,
                )

            # -- Translation settings --
            gr.Markdown("---\n**🌐 翻译设置（M2M100 CTranslate2）**")
            translate_checkbox = gr.Checkbox(
                label="启用翻译",
                value=user_settings.get("translate_enabled", False),
            )
            with gr.Row(equal_height=True) as translate_path_row:
                translation_model_path_box = gr.Textbox(
                    label="翻译模型存储路径",
                    value=user_settings.get("translation_model_path", DEFAULT_TRANSLATION_MODEL_DIR),
                    scale=3,
                    visible=user_settings.get("translate_enabled", False),
                )
                refresh_translation_btn = gr.Button(
                    "🔄",
                    variant="secondary",
                    scale=0,
                    min_width=40,
                    visible=user_settings.get("translate_enabled", False),
                )
                download_translation_btn = gr.Button(
                    "⬇ 下载翻译模型",
                    variant="secondary",
                    scale=1,
                    visible=_should_show_translation_download(user_settings),
                )
            with gr.Row(visible=user_settings.get("translate_enabled", False)) as translate_lang_row:
                target_lang_dropdown = gr.Dropdown(
                    choices=TARGET_LANGUAGE_CHOICES,
                    value=user_settings.get("target_lang", "zh"),
                    label="翻译为",
                    scale=1,
                    interactive=True,
                )
            translation_model_status = gr.Markdown(
                value=_translation_model_info(user_settings),
                visible=user_settings.get("translate_enabled", False),
            )

        with gr.Row():
            transcribe_btn = gr.Button(
                "▶ 开始转录",
                variant="primary",
                size="lg",
                interactive=False,
            )
            stop_btn = gr.Button(
                "⏹ 停止转录",
                variant="stop",
                size="lg",
                visible=False,
            )

        # ═══════════════════════════════════════════════════════════
        # Preview + Download + Status
        # ═══════════════════════════════════════════════════════════
        preview_box = gr.Textbox(
            label="实时转录预览",
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
            txt_download = gr.File(label="下载 TXT（纯文本）")
            srt_download = gr.File(label="下载 SRT（字幕）")

        status_md = gr.Markdown(value="就绪 — 请粘贴链接后点击 **分析**", elem_id="status_md")
        progress_area = gr.Markdown("", height=120)

        # ═══════════════════════════════════════════════════════════
        # Event handlers
        # ═══════════════════════════════════════════════════════════

        def _save_setting(**kw: str) -> None:
            """Persist one or more settings keys.  All settings.save() calls
            should go through this helper so persistence is easy to find."""
            settings.save(**kw)

        def on_model_select(model_size: str):
            path = whisper_model_path_box.value or DEFAULT_WHISPER_MODEL_DIR
            status = model_manager.list_models(path)
            s = status.get(model_size, {})
            _save_setting(model=model_size)
            return gr.update(visible=not s.get("downloaded"))

        def on_download_model(model_size: str, path: str,
                             progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_size:
                return (
                    gr.update(choices=_build_model_choices()),
                    "**❌ 未选择模型**",
                )
            try:
                model_manager.download_model(model_size, path or DEFAULT_WHISPER_MODEL_DIR)
            except OSError as e:
                logger.exception("Model download failed (disk/filesystem)")
                return (
                    gr.update(choices=_build_model_choices()),
                    f"**❌ 磁盘/文件系统错误：** {e}",
                )
            except Exception as e:
                logger.exception("Model download failed (network/unexpected)")
                return (
                    gr.update(choices=_build_model_choices()),
                    f"**❌ 下载失败（网络或未知错误）：** {e}",
                )
            return (
                gr.update(choices=_build_model_choices()),
                f"✅ **{model_size}** 下载完成！",
            )

        def on_save_device(device: str):
            _save_setting(device=device)

        def on_save_whisper_model_path(path: str):
            _save_setting(whisper_model_path=path)
            new_choices = _build_model_choices()
            # Also update download button: the currently selected model may
            # or may not exist at the new path.
            status = model_manager.list_models(path or DEFAULT_WHISPER_MODEL_DIR)
            model = settings.load().get("model", DEFAULT_MODEL)
            s = status.get(model, {})
            return (
                gr.update(choices=new_choices),
                gr.update(visible=not s.get("downloaded")),
            )

        # -- Translation handlers --

        def on_translate_checkbox(enabled: bool):
            _save_setting(translate_enabled=enabled)
            v = gr.update(visible=enabled)
            return v, v, v, v, v, v

        def on_save_target_lang(lang: str):
            _save_setting(target_lang=lang)

        def on_save_translation_model_path(path: str):
            _save_setting(translation_model_path=path)
            return _translation_model_info(settings.load())

        def on_refresh_translation_status():
            s = settings.load()
            return (
                gr.update(visible=_should_show_translation_download(s)),
                _translation_model_info(s),
            )

        def on_download_translation_model_btn(
            path: str,
            progress: gr.Progress = gr.Progress(track_tqdm=True),
        ):
            if not path:
                path = DEFAULT_TRANSLATION_MODEL_DIR
            _save_setting(translation_model_path=path)
            try:
                translation_model_manager.download_translation_model(path)
            except Exception as e:
                logger.exception("Translation model download failed")
                return (
                    gr.update(visible=True),
                    f"**❌ 下载失败：** {e}",
                )
            return (
                gr.update(visible=False),
                f"✅ 翻译模型下载完成！",
            )

        # ═══════════════════════════════════════════════════════════
        # Wire up events
        # ═══════════════════════════════════════════════════════════

        analyse_btn.click(
            fn=_analyse_video,
            inputs=[url_input],
            outputs=[analysis_md, part_selector, audio_selector, transcribe_btn, status_md],
            show_progress_on=progress_area,
        )

        model_dropdown.change(
            fn=on_model_select,
            inputs=[model_dropdown],
            outputs=[download_model_btn],
        )

        download_model_btn.click(
            fn=on_download_model,
            inputs=[model_dropdown, whisper_model_path_box],
            outputs=[model_dropdown, status_md],
            show_progress_on=progress_area,
        )

        device_radio.change(fn=on_save_device, inputs=[device_radio], outputs=[])
        whisper_model_path_box.change(
            fn=on_save_whisper_model_path,
            inputs=[whisper_model_path_box],
            outputs=[model_dropdown, download_model_btn],
        )

        refresh_models_btn.click(
            fn=_refresh_model_list,
            inputs=[whisper_model_path_box],
            outputs=[model_dropdown, download_model_btn],
        )

        demo.load(
            fn=_refresh_model_list,
            inputs=[whisper_model_path_box],
            outputs=[model_dropdown, download_model_btn],
        )
        demo.load(
            fn=on_refresh_translation_status,
            inputs=[],
            outputs=[download_translation_btn, translation_model_status],
        )

        language_dropdown.change(
            fn=lambda lang: _save_setting(language=lang),
            inputs=[language_dropdown],
            outputs=[],
        )

        transcribe_event = transcribe_btn.click(
            fn=_transcribe_pipeline,
            inputs=[
                url_input, model_dropdown, language_dropdown, device_radio,
                whisper_model_path_box,
                translate_checkbox, target_lang_dropdown,
                translation_model_path_box,
            ],
            outputs=[
                status_md, preview_box, summary_row,
                lang_md, duration_md, download_row,
                txt_download, srt_download, stop_btn, transcribe_btn,
            ],
            show_progress_on=progress_area,
        )

        stop_btn.click(
            fn=lambda: _current_stop_event[-1].set() if _current_stop_event else None,
        )

        # -- Translation events --
        translate_checkbox.change(
            fn=on_translate_checkbox,
            inputs=[translate_checkbox],
            outputs=[
                target_lang_dropdown, translate_lang_row,
                translation_model_path_box, refresh_translation_btn,
                download_translation_btn, translation_model_status,
            ],
        )

        target_lang_dropdown.change(
            fn=on_save_target_lang,
            inputs=[target_lang_dropdown],
            outputs=[],
        )

        translation_model_path_box.change(
            fn=on_save_translation_model_path,
            inputs=[translation_model_path_box],
            outputs=[translation_model_status],
        )

        refresh_translation_btn.click(
            fn=on_refresh_translation_status,
            inputs=[],
            outputs=[download_translation_btn, translation_model_status],
        )

        download_translation_btn.click(
            fn=on_download_translation_model_btn,
            inputs=[translation_model_path_box],
            outputs=[download_translation_btn, status_md],
            show_progress_on=progress_area,
        )

    return demo



# ======================================================================
# Launch
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    """Launch the Gradio WebUI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="vid2txt-webui",
        description="vid2txt Gradio WebUI",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config file (default: <project_root>/vid2txt_config.json).",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser on launch.",
    )
    args = parser.parse_args(argv)

    # Must happen before _build_ui() which calls settings.load()
    if args.config:
        settings.set_config_path(args.config)

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
        server_port=args.port,
        inbrowser=not args.no_browser,
        share=False,
        show_error=True,
        theme=gr.themes.Citrus(),
        css=_UI_CSS,
    )


if __name__ == "__main__":
    main()
