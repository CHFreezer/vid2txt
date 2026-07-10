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

---

## 🔴 HIGH — 代码质量审计 (2026-07-09, Round 2)

- [x] **[#13] webui.py — Stop 按钮在阻塞操作期间无响应**
  `src/webui.py:120,160` — `_transcribe_pipeline()` generator 在两个阶段无法响应停止事件：
  1. **下载阶段** (L120): `downloader(url)` 是阻塞调用，`stop_event.is_set()` 仅在进入转录循环后才检查
  2. **模型加载阶段** (L160 → `transcriber.py:198`): `transcribe_stream()` 内部调用 `_load_model()` 是阻塞的，用户点击停止后无反馈
  **现象**: 用户点击停止 → 事件被设置 → 但 generator 卡在阻塞调用中 → 直到操作完成才看到"转录已停止"→ 可能等待数分钟
  **修复方向**: 将下载和模型加载移到 generator 外部（在 `transcribe_btn.click` 之前完成），或使用 `threading` + 轮询 `stop_event` 在阻塞操作中定期检查

- [x] **[#14] transcriber.py + model_manager.py — `_REQUIRED_FILES` 常量重复定义**
  `src/transcriber.py:14` 和 `src/model_manager.py:17` 各自定义了完全相同的元组：
  ```python
  _REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
  ```
  **影响**: 若增加新的必需文件（如 `generation_config.json`），需要同时修改两处，易遗漏导致模型"完成性检查"不一致。
  **修复**: 将 `_REQUIRED_FILES` 移至 `src/config.py`，两处统一 `from .config import REQUIRED_MODEL_FILES`。

---

## 🟡 MEDIUM

- [x] **[#15] transcriber.py — `Transcriber.info` 在 batch `transcribe()` 后不更新**
  `src/transcriber.py:88-90,145-180` — `info` property 只在 `transcribe_stream()` 中赋值（L206-210），batch `transcribe()` 不会更新 `self._stream_info`。调用 `transcribe()` 后访问 `.info` 会得到上一次 `transcribe_stream()` 的残留数据或 `None`。
  **影响**: CLI 使用 batch API 不受影响（元数据从返回值获取），但若有人混合调用 batch + `.info` 会得到错误结果。
  **修复**: `transcribe()` 末尾同步更新 `self._stream_info`，或改用 `cached_property` 风格使 `.info` 返回最近一次转录（无论 batch/stream）的元数据。

- [x] **[#16] webui.py — `_analyse_video()` 使用 bare `except Exception`**
  `src/webui.py:316` — `except Exception as e:` 捕获所有异常并以相同方式处理，无法区分临时性错误（网络超时）和永久性错误（代码 bug）。
  **修复**: 拆分为 `except DownloadError` + `except (urllib.error.URLError, socket.timeout)` + 保留 bare `except Exception` 作为最后兜底并加 `logger.exception()`。

- [x] **[#17] test_regression.py — `test_stream_and_batch_produce_same_segments` 临时目录清理不在 finally 块**
  `tests/test_regression.py:286` — `cleanup_temp_dir(temp_dir)` 在函数末尾而非 `try/finally` 中。若 `transcriber2.transcribe_stream()` 抛出异常，temp_dir 泄漏。
  **修复**: 包裹 `try/finally` 确保清理。

- [x] **[#18] webui.py — 模型下载回调 `on_download_model` 使用 bare `except Exception`**
  `src/webui.py:583` — 下载失败时用户看到 "下载失败: {e}"，但异常信息可能不友好（如 `huggingface_hub` 的底层网络异常）。同时 `logger.exception()` 已记录完整堆栈，这是对的，但用户提示可以更友好（如区分网络错误 vs 磁盘空间不足）。
  **修复**: 捕获 `requests.HTTPError`、`OSError` 等具体异常并给出针对性提示。

---

## 🟢 LOW

- [x] **[#19] cli.py — epilog 未提及 `python -m src` 入口**
  `src/cli.py:46-48` — help 信息只写 `python main.py ...`，但项目支持 `python -m src`（通过 `__main__.py`）。README 也同样只提 `main.py`。
  **修复**: epilog 加一行 `python -m src https://...`。

- [x] **[#20] 缺少 ModelNotFoundError 路径的单元测试**
  `src/transcriber.py:101-108` — 当模型未下载时抛出 `ModelNotFoundError`，但没有测试覆盖此路径。虽然回归测试通过先下载模型来规避，但 CLI 的直接调用路径未被测试。
  **修复**: 新增 `tests/test_transcriber.py`，`test_raises_when_model_missing()` — 指向空临时目录创建 Transcriber 并断言 `ModelNotFoundError`。

- [x] **[#21] downloader.py — `download_audio()` glob 回退逻辑可能捡到错误文件**
  `src/downloader.py:140-154` — 第一轮 glob 排除 `.wav` 文件，第二轮 glob 匹配所有文件。若 temp_dir 中残留了上一次的音频文件（不应出现，因为有 `os.urandom(4).hex()` 唯一命名），会捡到旧文件。
  **影响**: 极低概率，`__call__()` 创建唯一 temp_dir 已基本杜绝。但代码意图不够清晰。
  **修复**: 添加注释说明逻辑，或将两轮合并为按 mtime 排序取最新文件。

- [x] **[#22] utils.py — `validate_url` 未覆盖 youtube.com/shorts 无 www 前缀**
  `tests/test_utils.py` — URL 验证测试有 `https://www.youtube.com/shorts/abc123def45` 但没有 `https://youtube.com/shorts/...`（无 www）。正则本身支持，只是测试未覆盖。
  **修复**: 在 parametrize 中加一条无 www 的 shorts URL。

- [x] **[#23] webui.py — `save` 调用散落各处，无统一入口**
  `src/webui.py:571,595,598-608,654` — settings.save() 通过 lambda、普通函数、change handler 等多种方式调用，逻辑分散。后续加新设置项时容易漏掉持久化。
  **修复**: 考虑封装一个 `_persist_setting(key, value)` helper，或利用 Gradio 的 `gr.State` 管理设置。

---

## 🚀 FEATURE — M2M100 CTranslate2 翻译功能 (2026-07-10)

> Hy-MT2 / llama-cpp-python 方案已废弃，最终采用 M2M100-418M + CTranslate2 int8 (468MB)。

### 前置重构：`model_path` → `whisper_model_path` (已完成 ✅)

> 翻译功能已实现并合并。以下为 2026-07-10 代码质量审计发现的问题。

---

## 🔴 代码质量审计 — 翻译功能上线后 (2026-07-10)

### BUG

- [x] **#24 webui.py — `on_translate_checkbox` 漏掉 `translation_model_status` 可见性**
  勾选"启用翻译"后状态文字（"✅ 翻译模型已就绪"）永远不会出现。
  **修复**: `on_translate_checkbox` 返回 6 个值（+ status），outputs 列表加 `translation_model_status`。

- [ ] **#25 webui.py — `detected_lang`/`detected_prob`/`audio_duration` 重复赋值**
  `_transcribe_pipeline` 中三行赋值代码出现了两次（246-248行），第二遍冗余覆盖。
  **修复**: 删除重复的三行。

- [ ] **#26 webui.py — 翻译 Row 容器未随 checkbox 隐藏**
  `translate_path_row` 包裹了路径输入框和按钮。子组件隐藏后 Row 本身留在页面上（空行）。
  **修复**: Row 加入 `on_translate_checkbox` 的 outputs 列表，随 checkbox 一起 toggle。

### 🟡 默认值分散（4 处重复定义）

- [ ] **#27 默认值唯一来源 — `"./models/faster-whisper"` 在 9 个地方硬编码**
  `settings.py`(×1) + `transcriber.py`(×1) + `webui.py`(×8)。应该有且仅有一个定义点。
  **修复**: `config.py` 新增 `DEFAULT_WHISPER_MODEL_DIR`，所有地方引用它。`settings._DEFAULTS` 也 import 这个常量。

- [ ] **#28 `"./models/m2m100"` 4 处硬编码未引用 `config.DEFAULT_TRANSLATION_MODEL_DIR`**
  `settings.py`、`translator.py`、`test_translation_model_manager.py` 写死字符串。
  **修复**: 统一 import `DEFAULT_TRANSLATION_MODEL_DIR`。

- [ ] **#29 `settings._DEFAULTS` 未 import config 常量**
  `"model": "base"` 应引用 `config.DEFAULT_MODEL`；`"language": "auto"` 散落多处。
  **修复**: settings.py import config 常量，消除重复字面量。

- [ ] **#30 `cli.py` `--target-lang` default 硬编码 `"zh"`**
  应引用 settings 默认值或 config 常量。
  **修复**: 从 `settings._DEFAULTS["target_lang"]` 取值或 import config 常量。

### 🟢 死代码 & 清理

- [ ] **#31 webui.py — 未使用的 `gr.State` (`translation_path_state`)**
  无任何事件读写，注释 "kept for compat" 无实际依赖。
  **修复**: 删除。

- [ ] **#32 webui.py — 未使用的 import `Generator`**
  `from typing import Generator` 未被任何类型注解引用。
  **修复**: 删除。

- [ ] **#33 settings.py — `load()` 中 `model_path → whisper_model_path` 迁移逻辑可移除**
  旧 key 已不存在于当前代码。保留会增加未来重命名负担。
  **修复**: 移除迁移代码块。

- [ ] **#34 translation_model_manager.py — `_ALLOW_PATTERNS` 比 `REQUIRED_TRANSLATION_MODEL_FILES` 多 `sentencepiece.bpe.model`**
  两个列表不同步，可能导致 is_model_downloaded 检查通过但 tokenizer 文件缺失。
  **修复**: 统一列表来源，或 `_ALLOW_PATTERNS` 改为 `list(REQUIRED_TRANSLATION_MODEL_FILES) + ["sentencepiece.bpe.model"]`。

### 🟢 测试改进

- [ ] **#35 test_regression.py — `_pipeline_result` 中 `try/finally` 太晚**
  `temp_dir` 清理在 `try` 块内，但 `transcriber.transcribe()` 在 `try` 之前。若转录抛异常，temp_dir 泄漏。
  **修复**: `try` 包裹整个函数体。

- [ ] **#36 test_regression.py — `test_stream_and_batch_produce_same_segments` 假设模型已下载**
  依赖 `_pipeline_result` fixture 先运行。单独运行此测试会 `ModelNotFoundError`。
  **修复**: 测试开头检查/下载模型。

- [ ] **#37 test_webui_regression.py — 访问私有属性 `settings._DEFAULT_CONFIG_PATH`**
  teardown 中 `settings.set_config_path(str(settings._DEFAULT_CONFIG_PATH))`。
  **修复**: 保存 `set_config_path` 前的路径，teardown 恢复。

### 🟢 设计改进

- [ ] **#38 cli.py — CLI 也支持 `-c/--config`**
  已添加参数但 default 仍硬编码，需统一从 settings 取值。
  **修复**: `Transcriber` 和 `Translator` 参数默认改为 `None`，内部 `settings.load()` 取值（参考 webui 逻辑）。

- [ ] **#39 Transcriber / Translator 默认参数应引用 config 而非写死**
  `Transcriber(whisper_model_path="./models/faster-whisper")` → 应 import config 常量。
  `Translator(model_path="./models/m2m100")` → 同上。
  **修复**: import 常量作为默认值。
