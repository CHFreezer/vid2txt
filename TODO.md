# TODO

- [x] **已修复: 停止转录按钮点击后 UI 不更新** (2026-07-09)
  `webui.py` — 删除 `stop_btn.click()` 中的 `cancels=[transcribe_event]`，让 generator 自然结束并完成 UI 更新。已通过 Playwright 实测验证。

---

## 🔴 HIGH — 代码质量审计 (2026-07-09)

- [x] **[#1] settings.py — `SETTINGS_FILE` 依赖 `cwd()` 导致路径不稳定**
  `src/settings.py:12` — 使用 `os.getcwd()` 构建配置文件路径，从不同目录启动 `webui.py` 会写入不同位置，用户设置看似"丢失"。
  **修复**: 改用基于 `__file__` 的项目根目录绝对路径 (`Path(__file__).resolve().parent.parent.parent`)。

- [x] **[#2] settings.py — `load()` / `save()` 存在 read-modify-write 竞态条件**
  `src/settings.py:38-48` — `save()` 先 `load()` 再写回，两个并发调用之间会丢失数据（如 CLI + WebUI 同时运行）。
  **修复**: 使用原子写入（写临时文件再 `os.replace`）。

- [x] **[#3] model_manager.py — 修改第三方库全局常量 `HF_HUB_DISABLE_XET`**
  `src/model_manager.py:60-61` — `download_model()` 直接设置 `huggingface_hub.constants.HF_HUB_DISABLE_XET = True`，污染进程级全局状态，后续任何 HF Hub 操作都会受影响。
  **修复**: 保存原值，`try/finally` 确保无论成功或异常都恢复。

---

## 🟡 MEDIUM

- [x] **[#4] cli.py — 未知异常错误地返回 `EXIT_DOWNLOAD`**
  `src/cli.py:193-198` — 通用 `except Exception` 捕获后返回 `EXIT_DOWNLOAD`（退出码 2），如果错误与下载无关（如配置文件损坏），调用者会得到误导信息。
  **修复**: 新增 `EXIT_UNKNOWN = 8` 错误码，通用异常使用新退出码。

- [x] **[#5] downloader.py — 未使用参数 + 裸 except + 冗余 import**
  `src/downloader.py`:
  - `download_audio()` 接收 `video_id` 参数但从未使用 (L119)
  - `__call__()` 的 except 块中重新 `import shutil`（L231），顶部 L5 已导入
  - `__call__()` 使用裸 `except Exception` (L229)，过于宽泛
  **修复**: 删除未使用参数和冗余 import，`except Exception` 改为 `except (DownloadError, ConversionError, OSError)`。

- [x] **[#6] downloader.py — `_check_ffmpeg()` 在不需要 ffmpeg 的场景中被调用**
  `src/downloader.py:32-36` — `Downloader.__init__` 无条件检查 ffmpeg，但 WebUI 的 `_analyse_video` 只调用 `get_all_parts_info()`（纯 yt-dlp），不需要 ffmpeg。
  **修复**: `__init__` 中移除强制检查，改为在 `convert_to_wav()` 首次调用时 lazy check。

- [x] **[#7] webui.py — 全局 `_stop_event` + `_hidden` 返回元组脆弱**
  `src/webui.py`:
  - L66: `_stop_event` 是模块级全局 `threading.Event`，Gradio queue 模式下多用户共享，A 用户点停止会终止 B 用户的转录
  - `_hidden()` 返回 10 元组，在 6 个 yield 点重复出现，增减 UI 组件时极易漏改
  **修复**: 使用 `_current_stop_event` 列表作 mutable cell，每次 pipeline 创建独立 Event；提取 `_HIDDEN_OUTPUTS` 常量，`_hidden` 通过解包复用。

- [x] **[#8] transcriber.py — 元数据通过私有属性泄露**
  `src/transcriber.py:128, webui.py:176-201` — `transcribe_stream()` 将语言/时长存为 `self._audio_duration` 等私有属性，调用者通过 `getattr(transcriber, "_audio_duration", 1)` 访问，耦合脆弱。
  **修复**: 新增 `TranscriptionInfo` dataclass + `Transcriber.info` property，WebUI 改用 `transcriber.info.language` 等公共接口。

- [x] **[#9] cuda_setup.py — CUDA DLL 文件名硬编码版本号**
  `src/cuda_setup.py:62` — 预加载列表写死 `cudart64_12.dll` / `cublas64_12.dll` / `cublasLt64_12.dll`，升级 CUDA 13 后这些文件名会变成 `*_13.dll`，预加载静默失败。
  **修复**: 改用 glob 模式匹配 `cudart64_*.dll` 等，适配任意 CUDA 主版本。

---

## 🟢 LOW

- [x] **[#10] downloader.py — `get_all_parts_info` 静默忽略 JSON 解析错误**
  `src/downloader.py:112-113` — `except json.JSONDecodeError: pass` 静默丢弃非 JSON 行，如果 yt-dlp 输出异常，用户得不到任何警告，可能导致静默丢失分P信息。
  **修复**: 添加 `logger.debug()` 记录被跳过的行。

- [x] **[#11] 代码清理 — 未使用 import + 变量命名 + 冗余赋值**
  - `settings.py:10` — `from dataclasses import dataclass` 未使用 → **已删除**
  - `cli.py:151` — `word_count` 变量名误导 → **改为 `char_count`**
  - `cli.py:133-136` — 判断 `result["segments"]` 为空后又重复赋值 → **删除冗余赋值**

- [x] **[#12] 零测试覆盖**
  项目无任何自动化测试。
  **修复**: 引入 pytest，新增 `tests/test_utils.py`（22 个用例）和 `tests/test_formatter.py`（8 个用例），覆盖 URL 校验、时间戳格式化、文件名清洗、TXT/SRT 格式化。**36/36 全部通过**。
