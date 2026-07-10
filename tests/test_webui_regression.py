"""WebUI end-to-end regression tests via Playwright.

Launches the Gradio WebUI in a background thread, navigates with a real
browser, and verifies the two-step workflow (Analyse -> Transcribe) against
three fixed videos.

Strategy
--------
- **Smoke**: page loads, header visible, core controls present.
- **Analyse**: enter each URL, click Analyse, verify metadata appears.
- **Settings**: model/device/language controls visible and interactive.
- **Full pipeline** (slow): YouTube Shorts -- URL -> Analyse -> Transcribe ->
  verify preview text, download files, and summary stats.
- **Stop**: verify the stop button interrupts a running transcription.

Slow tests are marked ``@pytest.mark.slow``.
"""

import os
import sys
import time
import threading

import pytest

from src import settings

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PORT = 7860
BASE_URL = f"http://127.0.0.1:{PORT}"

BILIBILI_URL = "https://www.bilibili.com/video/BV1Kt7q6hEAr"
YOUTUBE_URL  = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
SHORTS_URL   = "https://www.youtube.com/shorts/Lp1o_IDZ7vk"


# =============================================================================
# Server fixture (in-process thread)
# =============================================================================

@pytest.fixture(scope="session")
def webui_server(tmp_path_factory):
    """Start Gradio WebUI in a background daemon thread, yield the base URL,
    and clean up after all tests complete.

    Uses a temporary config file so tests don't pollute the real
    ``vid2txt_config.json``.
    """
    import json as _json

    # Create a clean test config
    test_config = tmp_path_factory.mktemp("vid2txt_test") / "config.json"
    test_config.write_text(_json.dumps({
        "device": "cpu",
        "whisper_model_path": "./models/faster-whisper",
        "model": "tiny",
        "language": "auto",
        "translate_enabled": False,
        "target_lang": "zh",
        "translation_model": "1.8B-1.25Bit",
        "translation_model_path": "./models/hy-mt2",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    settings.set_config_path(str(test_config))

    from src.webui import _build_ui

    demo = _build_ui()

    # Use a threading.Event to signal when the server is up
    ready = threading.Event()
    error: list[Exception] = []

    def _start() -> None:
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=PORT,
                inbrowser=False,
                share=False,
                show_error=True,
                quiet=True,
            )
        except Exception as e:
            error.append(e)
            ready.set()

    # Override Gradio's blocking behaviour: launch in a daemon thread
    thread = threading.Thread(target=_start, daemon=True)
    thread.start()

    # Wait for the server to accept connections
    import urllib.request
    import urllib.error

    deadline = time.time() + 60
    while time.time() < deadline:
        if error:
            pytest.fail(f"Gradio failed to start: {error[0]}")
        try:
            urllib.request.urlopen(f"{BASE_URL}/", timeout=2)
            break
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(1)
    else:
        pytest.fail("WebUI did not start within 60s")

    time.sleep(1)
    yield BASE_URL

    # Teardown
    demo.close()
    # Restore default config path so subsequent imports use the real one
    settings.set_config_path(str(settings._DEFAULT_CONFIG_PATH))


# =============================================================================
# Page helpers
# =============================================================================

def _goto(page, webui_server: str) -> None:
    """Navigate to the WebUI and wait for Gradio to render."""
    page.goto(webui_server, wait_until="domcontentloaded")
    page.wait_for_selector("gradio-app", timeout=15_000)
    page.wait_for_timeout(500)


def _url_input(page):
    """Return the URL textbox locator."""
    return page.get_by_placeholder("粘贴视频链接...（Bilibili / YouTube / Shorts）")


def _analyse_btn(page):
    """Return the Analyse button locator."""
    return page.get_by_role("button", name="🔍 分析")


def _transcribe_btn(page):
    """Return the Transcribe button locator."""
    return page.get_by_role("button", name="▶ 开始转录")


def _stop_btn(page):
    """Return the Stop button locator."""
    return page.get_by_role("button", name="⏹ 停止转录")


def _wait_for_status(page, text_contains: str, timeout: int = 30_000) -> None:
    """Wait until any .prose or <p> element contains *text_contains*."""
    page.wait_for_function(
        f"""
        () => {{
            const els = document.querySelectorAll('.prose, p');
            return Array.from(els).some(
                el => el.textContent && el.textContent.includes({text_contains!r})
            );
        }}
        """,
        timeout=timeout,
    )


# =============================================================================
# Smoke
# =============================================================================

class TestSmoke:
    """Verify the page loads and renders core elements."""

    @pytest.fixture(autouse=True)
    def _navigate(self, page, webui_server: str) -> None:
        _goto(page, webui_server)

    def test_page_loads(self, page) -> None:
        assert page.locator("gradio-app").count() > 0

    def test_header_visible(self, page) -> None:
        heading = page.get_by_role("heading", name="🎤 vid2txt")
        assert heading.is_visible()

    def test_url_input_visible(self, page) -> None:
        assert _url_input(page).is_visible()

    def test_analyse_button_visible(self, page) -> None:
        assert _analyse_btn(page).is_visible()

    def test_transcribe_button_starts_disabled(self, page) -> None:
        assert _transcribe_btn(page).is_disabled()

    def test_status_shows_ready(self, page) -> None:
        status = page.get_by_text("就绪")
        assert status.is_visible()


# =============================================================================
# Analyse (Step 1 -- metadata)
# =============================================================================

class TestAnalyse:
    """Enter each video URL, click Analyse, and verify metadata appears."""

    @pytest.fixture(autouse=True)
    def _navigate(self, page, webui_server: str) -> None:
        _goto(page, webui_server)

    def _analyse_url(self, page, url: str) -> None:
        _url_input(page).fill(url)
        _analyse_btn(page).click()
        page.wait_for_timeout(3000)
        _wait_for_status(page, "分析完成", timeout=30_000)

    def test_analyse_bilibili(self, page) -> None:
        self._analyse_url(page, BILIBILI_URL)
        assert page.get_by_text("📺").is_visible()

    def test_analyse_youtube(self, page) -> None:
        self._analyse_url(page, YOUTUBE_URL)
        assert page.get_by_text("📺").is_visible()

    def test_analyse_shorts(self, page) -> None:
        self._analyse_url(page, SHORTS_URL)
        assert page.get_by_text("📺").is_visible()

    def test_transcribe_btn_enabled_after_analysis(self, page) -> None:
        _url_input(page).fill(SHORTS_URL)
        _analyse_btn(page).click()
        page.wait_for_timeout(3000)
        _wait_for_status(page, "分析完成", timeout=30_000)
        assert _transcribe_btn(page).is_enabled()


# =============================================================================
# Settings panel
# =============================================================================

class TestSettings:
    """Verify settings controls exist and are functional."""

    @pytest.fixture(autouse=True)
    def _navigate(self, page, webui_server: str) -> None:
        _goto(page, webui_server)

    def test_settings_accordion_visible(self, page) -> None:
        accordion = page.get_by_text("模型与设备设置")
        assert accordion.is_visible()

    def test_whisper_model_path_input_visible(self, page) -> None:
        mp = page.get_by_label("模型存储路径")
        assert mp.is_visible()
        assert mp.input_value() == "./models/faster-whisper"

    def test_model_dropdown_visible(self, page) -> None:
        combo = page.get_by_role("combobox", name="Whisper 模型")
        assert combo.is_visible()
        val = combo.input_value()
        assert val, "Model dropdown should have a selected value"

    def test_language_dropdown_visible(self, page) -> None:
        combo = page.get_by_role("combobox", name="语言")
        assert combo.is_visible()
        assert combo.input_value(), "Language dropdown should have a value"

    def test_device_radio_visible(self, page) -> None:
        cpu_radio = page.get_by_role("radio", name="💻 CPU")
        cuda_radio = page.get_by_role("radio", name="⚡ CUDA")
        assert cpu_radio.is_visible()
        assert cuda_radio.is_visible()
        assert cpu_radio.is_checked() or cuda_radio.is_checked()

    @pytest.mark.slow
    def test_download_model_button(self, page, tmp_path) -> None:
        """Download a model to a temp directory via the WebUI, verify
        files, then clean up — without touching the real whisper_model_path."""
        import shutil

        original_settings = settings.load()
        original_model_path = original_settings.get("whisper_model_path", "./models/faster-whisper")
        original_model = original_settings.get("model", "base")
        temp_model_dir = str(tmp_path / "test_models")

        whisper_model_path_input = page.get_by_label("模型存储路径")
        download_btn = page.get_by_role("button", name="⬇ 下载模型")
        model_combo = page.get_by_role("combobox", name="Whisper 模型")

        def _click_option(text_fragment: str) -> None:
            """Click a dropdown option by partial text match."""
            model_combo.click()
            page.wait_for_timeout(500)
            option = page.locator('[role="option"]').filter(has_text=text_fragment)
            option.wait_for(state="visible", timeout=5_000)
            option.evaluate("el => el.click()")
            page.wait_for_timeout(1000)

        try:
            # -- Step 1: Switch whisper_model_path to temp dir --
            whisper_model_path_input.click()
            page.keyboard.press("Control+a")
            page.keyboard.press("Backspace")
            page.keyboard.type(temp_model_dir)
            page.locator("h1").click()  # blur -> on_save_whisper_model_path
            page.wait_for_timeout(2000)

            # Verify whisper_model_path change propagated (combobox labels update,
            # and download button appears for current model at empty path)
            val = model_combo.input_value()
            assert "[未下载]" in val, f"Model path change didn't apply: {val}"
            assert download_btn.is_visible(), (
                "Download button should appear after switching to empty path"
            )

            # -- Step 2: Select tiny and download --
            _click_option("tiny")
            download_btn.click()
            _wait_for_status(page, "下载完成", timeout=180_000)

            # -- Step 3: Verify model files on disk --
            model_dir = os.path.join(temp_model_dir, "faster-whisper-tiny")
            assert os.path.isdir(model_dir), f"Model dir not found: {model_dir}"
            for f in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
                fp = os.path.join(model_dir, f)
                assert os.path.isfile(fp), f"Missing file: {fp}"

        finally:
            # -- Step 4: Restore whisper_model_path to original --
            whisper_model_path_input.click()
            page.keyboard.press("Control+a")
            page.keyboard.press("Backspace")
            page.keyboard.type(original_model_path)
            page.locator("h1").click()
            page.wait_for_timeout(1000)

            # Restore model selection
            _click_option(original_model)

            # -- Step 5: Clean up temp directory --
            shutil.rmtree(temp_model_dir, ignore_errors=True)


# =============================================================================
# Full pipeline (slow) -- Analyse -> Transcribe -> Verify output
# =============================================================================

@pytest.mark.slow
class TestFullPipeline:
    """Full two-step workflow: analyse the Shorts video, transcribe with the
    ``tiny`` model, then verify preview text and download buttons appear."""

    @pytest.fixture(autouse=True)
    def _navigate(self, page, webui_server: str) -> None:
        _goto(page, webui_server)

    def test_full_transcribe_pipeline(self, page) -> None:
        # ---- Step 1: Select tiny model for speed ----
        model_combo = page.get_by_role("combobox", name="Whisper 模型")
        if not model_combo.is_visible():
            page.get_by_text("模型与设备设置").click()
            page.wait_for_timeout(500)
        model_combo.click()
        page.wait_for_timeout(500)
        option = page.locator('[role="option"]').filter(has_text="tiny")
        option.wait_for(state="visible", timeout=5_000)
        option.click()
        page.wait_for_timeout(500)

        # ---- Step 2: Analyse ----
        _url_input(page).fill(SHORTS_URL)
        _analyse_btn(page).click()
        page.wait_for_timeout(3000)
        _wait_for_status(page, "分析完成", timeout=30_000)

        # ---- Step 3: Transcribe ----
        _transcribe_btn(page).click()
        _wait_for_status(page, "转录完成", timeout=300_000)

        # ---- Step 4: Verify preview text ----
        preview_box = page.locator("textarea").last
        preview_text = preview_box.input_value()
        assert len(preview_text) > 0, "Preview text is empty after transcription"

        # ---- Step 5: Verify download files ----
        page.wait_for_selector("text=下载 TXT", timeout=10_000)
        page.wait_for_selector("text=下载 SRT", timeout=10_000)

        # ---- Step 6: Verify summary ----
        # Summary stats appear as Markdown (language, duration, segments, chars)
        page.wait_for_timeout(1000)
        # The summary info is in the status_md component which uses .prose
        all_prose = page.locator(".prose")
        combined = "\n".join(
            (el.text_content() or "") for el in all_prose.all()
        )
        assert "语言" in combined, f"Language stat missing from: {combined[:200]}"

    def test_stop_button_works(self, page) -> None:
        # -- Ensure Gradio queue is idle after previous test --
        page.wait_for_timeout(2000)

        # -- Select tiny model --
        model_combo = page.get_by_role("combobox", name="Whisper 模型")
        if not model_combo.is_visible():
            page.get_by_text("模型与设备设置").click()
            page.wait_for_timeout(500)
        model_combo.click()
        page.wait_for_timeout(500)
        option = page.locator('[role="option"]').filter(has_text="tiny")
        option.wait_for(state="visible", timeout=5_000)
        option.click()
        page.wait_for_timeout(500)

        # -- Analyse --
        _url_input(page).fill(SHORTS_URL)
        _analyse_btn(page).click()
        page.wait_for_timeout(3000)
        _wait_for_status(page, "分析完成", timeout=30_000)

        # -- Start transcription --
        _transcribe_btn(page).click()

        # Stop button only appears during transcription phase (after download
        # + model loading). Wait generously.
        _stop_btn(page).wait_for(state="visible", timeout=120_000)

        # -- Click stop --
        _stop_btn(page).click()
        _wait_for_status(page, "停止", timeout=30_000)
