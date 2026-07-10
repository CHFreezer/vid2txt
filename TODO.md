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

## 🚀 FEATURE — Hy-MT2 翻译功能 (2026-07-10)

### 前置重构：`model_path` → `whisper_model_path`

在加翻译功能之前，先把现有 Whisper 模型路径的命名明确语义，与新翻译模型路径对称。

**逐文件改动清单：**

| 文件 | 改动 |
|------|------|
| `src/settings.py:18` | `_DEFAULTS["model_path"]` → `_DEFAULTS["whisper_model_path"]`，值 `"./models/faster-whisper"` |
| `src/settings.py:36` | `save(model_path=...)` → `save(whisper_model_path=...)` |
| `src/settings.py:42-43` | `if model_path is not None: current["model_path"] = model_path` → `whisper_model_path` |
| `src/settings.py` 新增 | `load()` 中兼容旧 key：若 JSON 中存在 `model_path` 但没有 `whisper_model_path`，自动迁移 |
| `src/model_manager.py:39` | `list_models(model_path)` → `list_models(whisper_model_path)` |
| `src/model_manager.py:56` | `download_model(size, model_path)` → `download_model(size, whisper_model_path)` |
| `src/model_manager.py:28` | 内部函数 `_custom_model_path` 参数名 `base`，不改（语义已明确） |
| `src/transcriber.py:70` | `__init__(model_path="./models")` → `__init__(whisper_model_path="./models/faster-whisper")` |
| `src/transcriber.py:80` | `self.model_path = model_path` → `self.whisper_model_path = whisper_model_path` |
| `src/transcriber.py:59,92,98,105` | 所有 `self.model_path` 引用 → `self.whisper_model_path` |
| `src/webui.py:527-529` | Gradio 组件 `model_path_box` → `whisper_model_path_box`，`value` 默认值同步更新 |
| `src/webui.py:439,481,615` | 局部变量 `model_path` → `whisper_model_path` |
| `src/webui.py:650-651` | `on_save_model_path` → `on_save_whisper_model_path`，`_save_setting(model_path=...)` → `whisper_model_path` |
| `src/webui.py:682,688-690,696,702,714` | 所有 `model_path_box` 引用 → `whisper_model_path_box` |
| `src/cli.py` | 如有引用 `model_path` 处，同步改名 |
| `tests/` | 所有测试中的 `model_path` → `whisper_model_path` |

**兼容处理：** `settings.load()` 检测旧 key `model_path` 存在而 `whisper_model_path` 不存在时，自动将旧值迁移到新 key 并写回 `vid2txt_config.json`。

**用户操作：** 手动将 `./models/faster-whisper-*` 目录移动到 `./models/faster-whisper/` 下。

**最终目录结构：**
```
./models/
├── faster-whisper/              ← whisper_model_path
│   ├── faster-whisper-base/
│   ├── faster-whisper-large-v3/
│   └── ...
└── hy-mt2/                      ← translation_model_path
    ├── Hy-MT2-1.8B-Q4_K_M.gguf
    └── ...
```

---

### 技术选型

| 项 | 选择 |
|------|------|
| 模型 | `tencent/Hy-MT2` GGUF 系列（详见下方"可选模型"表） |
| 推理引擎 | `llama-cpp-python` + CUDA |
| 翻译范围 | 33 语言（覆盖 WebUI 现有全部语言选项） |
| 和现有栈关系 | CTranslate2 (转录) + llama.cpp (翻译)，双引擎串行，不冲突 |
| 翻译机制 | **可选** — 默认关闭，用户手动开启 |

### 可选模型（官方 GGUF）

| 文件 | 体积 | 质量 | 推荐场景 |
|------|------|------|---------|
| `Hy-MT2-1.8B-Q4_K_M.gguf` | 1.13 GB | ⭐⭐⭐ | 🏆 **默认推荐** — 最佳性价比 |
| `Hy-MT2-1.8B-Q6_K.gguf` | 1.47 GB | ⭐⭐⭐½ | 偏质量 |
| `Hy-MT2-1.8B-Q8_0.gguf` | 1.91 GB | ⭐⭐⭐⭐ | 1.8B 最高质量 |
| `Hy-MT2-7B-Q4_K_M.gguf` | 4.62 GB | ⭐⭐⭐⭐½ | 7B 入门 |
| `Hy-MT2-7B-Q6_K.gguf` | 6.16 GB | ⭐⭐⭐⭐⭐ | 高配 |
| `Hy-MT2-7B-Q8_0.gguf` | 7.98 GB | ⭐⭐⭐⭐⭐ | 顶配 |

> 全部来自 `tencent/Hy-MT2-1.8B-GGUF` 和 `tencent/Hy-MT2-7B-GGUF`

### 可翻译的目标语言

**独立于 Whisper 转录语言**。这是 Hy-MT2 原生支持的 33 种目标语言（Hy-MT2 的 prompt 使用短代码）：

| 显示名称 | 代码 | | 显示名称 | 代码 |
|---------|------|-|---------|------|
| 中文 (简体) | `zh` | 🏆 | 中文 (繁體) | `zh-Hant` |
| English | `en` | | 日本語 | `ja` |
| 한국어 | `ko` | | Français | `fr` |
| Deutsch | `de` | | Español | `es` |
| Русский | `ru` | | ไทย | `th` |
| Tiếng Việt | `vi` | | Português | `pt` |
| Türkçe | `tr` | | العربية | `ar` |
| Italiano | `it` | | Bahasa Melayu | `ms` |
| Bahasa Indonesia | `id` | | Filipino | `tl` |
| हिन्दी | `hi` | | Polski | `pl` |
| Čeština | `cs` | | Nederlands | `nl` |
| ភាសាខ្មែរ | `km` | | မြန်မာဘာသာ | `my` |
| فارسی | `fa` | | ગુજરાતી | `gu` |
| اردو | `ur` | | తెలుగు | `te` |
| मराठी | `mr` | | עברית | `he` |
| বাংলা | `bn` | | தமிழ் | `ta` |
| Українська | `uk` | | | |

> 共 33 个选项，默认中文(简体) `zh`。代码来自 Hy-MT2 官方 README。**与转录语言下拉框 `LANGUAGE_CHOICES` 完全独立。**

### 翻译提示词（Chat Format）

Hy-MT2 不使用 system prompt，直接用 user message：

```
Translate from {source_lang_name} to {target_lang_name}:
{segment_text}
```

### 流水线

```
下载音频 → 转录 Whisper → [可选] 翻译 Hy-MT2 → 格式化输出
```

翻译阶段仅在用户启用时执行。未启用时流水线与现有完全相同（零开销）。

---

### 任务列表

- [ ] **Task 1: 项目配置 & 依赖**
  - `src/config.py` — 新增翻译相关**常量**（不可变，类比 `SUPPORTED_MODELS`）
    - `TARGET_LANGUAGE_CHOICES` — Hy-MT2 原生支持的 33 种目标语言 (`list[tuple[str, str]]`)，**独立于转录语言 `LANGUAGE_CHOICES`**
      ```python
      TARGET_LANGUAGE_CHOICES = [
          ("中文(简体)", "zh"), ("中文(繁體)", "zh-Hant"),
          ("English", "en"), ("日本語", "ja"),
          ("한국어", "ko"), ("Français", "fr"),
          ("Deutsch", "de"), ("Español", "es"),
          ("Русский", "ru"), ("ไทย", "th"),
          ("Tiếng Việt", "vi"), ("Português", "pt"),
          ("Türkçe", "tr"), ("العربية", "ar"),
          ("Italiano", "it"), ("Bahasa Melayu", "ms"),
          ("Bahasa Indonesia", "id"), ("Filipino", "tl"),
          ("हिन्दी", "hi"), ("Polski", "pl"),
          ("Čeština", "cs"), ("Nederlands", "nl"),
          ("ភាសាខ្មែរ", "km"), ("မြန်မာဘာသာ", "my"),
          ("فارسی", "fa"), ("ગુજરાતી", "gu"),
          ("اردو", "ur"), ("తెలుగు", "te"),
          ("मराठी", "mr"), ("עברית", "he"),
          ("বাংলা", "bn"), ("தமிழ்", "ta"),
          ("Українська", "uk"),
      ]
      ```
    - `TRANSLATION_MODEL_REPOS` — 6 个 GGUF 文件的 repo/filename 映射
      ```python
      TRANSLATION_MODEL_REPOS = {
          "1.8B-Q4_K_M": {"repo": "tencent/Hy-MT2-1.8B-GGUF", "filename": "Hy-MT2-1.8B-Q4_K_M.gguf", "size_gb": 1.13},
          "1.8B-Q6_K":   {"repo": "tencent/Hy-MT2-1.8B-GGUF", "filename": "Hy-MT2-1.8B-Q6_K.gguf",   "size_gb": 1.47},
          "1.8B-Q8_0":   {"repo": "tencent/Hy-MT2-1.8B-GGUF", "filename": "Hy-MT2-1.8B-Q8_0.gguf",   "size_gb": 1.91},
          "7B-Q4_K_M":   {"repo": "tencent/Hy-MT2-7B-GGUF",   "filename": "Hy-MT2-7B-Q4_K_M.gguf",   "size_gb": 4.62},
          "7B-Q6_K":     {"repo": "tencent/Hy-MT2-7B-GGUF",   "filename": "HY-MT2-7B-Q6_K.gguf",     "size_gb": 6.16},
          "7B-Q8_0":     {"repo": "tencent/Hy-MT2-7B-GGUF",   "filename": "HY-MT2-7B-Q8_0.gguf",     "size_gb": 7.98},
      }
      ```
    - `SUPPORTED_TRANSLATION_MODELS = tuple(TRANSLATION_MODEL_REPOS.keys())`
    - `DEFAULT_TRANSLATION_MODEL_DIR = "./models/hy-mt2"` — 翻译模型存储根目录
    - `TRANSLATION_INFERENCE_PARAMS` — 推理超参（temperature/top_p/top_k/repetition_penalty/max_tokens）
  - `src/settings.py` — 新增翻译相关**用户偏好**（可变，类比 `model`、`language`）
    - `_DEFAULTS` 新增 4 个 key:
      ```python
      "translate_enabled": False,
      "target_lang": "zh",      # 首次默认简体中文
      "translation_model": "1.8B-Q4_K_M",
      "translation_model_path": "./models/hy-mt2",
      ```
    - `save()` 签名扩展 4 个 keyword 参数:
      ```python
      def save(device=None, ..., target_lang=None, translation_model=None):
      ```
      每个参数 `is not None` 时写入 `current`，与现有 `device`/`model` 等完全一致
  - `requirements.txt` — 添加 `llama-cpp-python`
  - `CLAUDE.md` — 双引擎架构文档 + CUDA llama-cpp 安装说明

- [ ] **Task 2: 翻译模型管理 `src/translation_model_manager.py`**
  - 复用 `src/model_manager.py` 的下载模式（逐文件 + tqdm 进度条 + Xet 禁用/恢复）
  - 函数签名均从 `config.py` 读取常量，遵循现有 `model_manager.py` 风格
  - `list_translation_models(model_path)` → `dict[str, dict]`
    - 遍历 `SUPPORTED_TRANSLATION_MODELS`，检查每个 GGUF 文件是否存在
    - 返回 `{"1.8B-Q4_K_M": {"downloaded": True/False, "path": "...", "size_gb": 1.13}, ...}`
  - `download_translation_model(model_key, model_path)` → `str`
    - `model_key` 查 `TRANSLATION_MODEL_REPOS` 得到 repo + filename
    - 单文件下载（非 snapshot_download），复用 Xet 禁用/恢复逻辑
    - 下载到 `{model_path}/hy-mt2/{filename}`
  - `get_model_path(model_key, base_path)` → `Path`
  - 模型存储结构:
    ```
    ./models/hy-mt2/
      Hy-MT2-1.8B-Q4_K_M.gguf
      Hy-MT2-1.8B-Q6_K.gguf
      Hy-MT2-7B-Q4_K_M.gguf
      ...
    ```

- [ ] **Task 3: 翻译器 `src/translator.py`**
  - `class Translator`:
    - `__init__(model_path: str, device: str = "cpu", n_gpu_layers: int = 0)`
      - CUDA: `n_gpu_layers = -1`（全部 GPU）
      - CPU: `n_gpu_layers = 0`
    - `_load_model()` — lazy load GGUF via `llama_cpp.Llama`
      - 推理参数从 `config.TRANSLATION_INFERENCE_PARAMS` 读取
    - `translate(text: str, source_lang: str, target_lang: str)` → `str`
      - 单段翻译，chat format
    - `translate_segments(segments: list[Segment], source_lang: str, target_lang: str)` → `list[Segment]`
      - 逐段翻译，保留 `start`/`end` 时间戳
      - 返回的 `Segment` 中 `text` = 原文，新增 `translated_text` = 译文
      - 流式友好：每翻译完一段 yield 一条 `Segment`
    - `translate_segments_stream(segments, ...)` → `Generator[Segment]`
      - 逐段 yield，适配 WebUI 实时预览
    - `unload()` — 释放模型，供流水线阶段切换
    - `is_loaded` — property
  - 类型扩展:
    ```python
    class Segment(TypedDict):
        start: float
        end: float
        text: str
        translated_text: str | None  # 新增 — 译文，未翻译时为 None
    ```

- [ ] **Task 4: CLI 集成 `src/cli.py`**
  - 新增 CLI 参数:
    - `--translate` — 启用翻译（flag，默认不启用）
    - `--target-lang` / `-t` — 目标语言代码，仅 `--translate` 时有效 (default: `zh`)
    - `--translation-model` — 模型选择 (choices: 6 个 key，default: `1.8B-Q4_K_M`)
    - `--translation-model-path` — 模型目录 (default: `./models/hy-mt2`)
  - 流水线改为:
    ```
    Phase 1: 下载音频
    Phase 2: 转录 (Whisper)
    Phase 3: 翻译 (Hy-MT2) ← 仅在 --translate 时执行
    Phase 4: 格式化输出
    ```
  - 输出文件命名:
    - 无翻译: `{basename}.txt` / `{basename}.srt`（不变）
    - 有翻译: `{basename}.{target_lang}.txt` / `{basename}.{target_lang}.srt`
    - 双语文件: `{basename}.bilingual.txt`（原文+译文对照）
  - 新增退出码: `EXIT_TRANSLATION_MODEL = 9`
  - 错误处理: 模型未下载 → 提示 `python -c "from src.translation_model_manager import download_translation_model; download_translation_model('1.8B-Q4_K_M', './models/hy-mt2')"`

- [ ] **Task 5: Formatter 扩展 `src/formatter.py`**
  - 新增 `to_bilingual_txt(segments)` — 原文 + 译文对照，双行格式:
    ```
    [00:05 → 00:10]
    你好世界，今天天气很好。
    Hello World, the weather is great today.
    
    [00:11 → 00:18]
    我们开始吧。
    Let's get started.
    ```
  - 新增 `to_translated_txt(segments)` — 纯译文
  - 新增 `to_translated_srt(segments)` — 译文字幕（时间戳不变）
  - 新增 `to_bilingual_srt(segments)` — 双语字幕:
    ```
    1
    00:00:05,000 --> 00:00:10,000
    你好世界，今天天气很好。
    Hello World, the weather is great today.
    ```
  - 保留现有 `to_txt` / `to_srt`（不含翻译时用原文，行为不变）
  - `write()` 扩展:
    ```python
    def write(segments, output_dir, basename, translated=False, target_lang=None):
        # 始终输出原文 TXT/SRT
        # 如有翻译，额外输出译文 TXT/SRT + 双语 TXT
    ```

- [ ] **Task 6: WebUI 集成 `src/webui.py`**
  - **Settings accordion 新增区域** "🌐 翻译设置":
    - `translate_enabled_checkbox` — "启用翻译"（默认关闭）
    - `target_lang_dropdown` — 目标语言选择，label 如 "翻译为"
      - 选项：33 种 Hy-MT2 目标语言（与 WebUI 转录语言下拉框**独立**，两套不同的语言集合）
      - 默认: 中文(简体) `zh`；之后**记住用户最后一次选择**（通过 `settings.save()` 持久化 `target_lang`）
      - 仅在勾选"启用翻译"后可见（`interactive` + `visible` 联动）
    - `translation_model_dropdown` — 模型选择
      - 选项: 从 `list_translation_models()` 动态生成，带 `[已下载]` / `[未下载]` 标记
      - 默认 `1.8B-Q4_K_M`
    - `download_translation_btn` — "⬇ 下载翻译模型"
      - 仅在所选模型未下载时可见
      - 下载进度条锚定到 `progress_area`
    - `translation_model_status_md` — 显示所选模型体积、推荐说明
  - **预览区域改造** — 单框双语显示:
    - `preview_box` 保持为单一 `gr.Textbox`，**不新增第二个框**
    - 翻译关闭时: 行为与现在完全一致，只显示原文
    - 翻译开启时: 每段追加两行，实时交替:
      ```
      [00:00:05 → 00:00:10]
      🎙 你好世界，今天天气很好。
      🌐 Hello World, the weather is great today.
      
      [00:00:11 → 00:00:18]
      🎙 我们开始吧。
      🌐 Let's get started.
      ```
    - 图标区分原文(🎙)和译文(🌐)，一条 segment 的原文和译文同时出现
    - 流式翻译: 转录完一段 → 立刻翻译 → 原文+译文一起追加到预览框
  - **输出下载区扩展**:
    - 翻译关闭时: 只显示 TXT + SRT 下载（不变）
    - 翻译开启时: 显示 TXT(原文) + SRT(原文) + TXT(译文) + SRT(译文) + TXT(双语)
  - **流水线 `_transcribe_pipeline`** 逻辑:
    ```
    if translate_enabled:
        ① 下载音频 → ② 转录(流式) → ③ 翻译(逐段,流式) → ④ 格式化
        预览: 转录一段 → 立即翻译 → 原文+译文追加
    else:
        ① 下载音频 → ② 转录(流式) → ④ 格式化（现有逻辑，不动）
    ```
  - **Stop 按钮**: 同样中断翻译阶段
  - **Settings 持久化**:
    - `translate_enabled: bool`
    - `target_lang: str`
    - `translation_model: str` — 如 `"1.8B-Q4_K_M"`

- [ ] **Task 7: 翻译偏好持久化 `src/settings.py`**
  - 在现有 `vid2txt_config.json` 基础上扩展，遵循已有模式
  - `_DEFAULTS` 新增 4 个 key（与现有 `device`/`model`/`language` 同级）:
    ```python
    "translate_enabled": False,
    "target_lang": "zh",
    "translation_model": "1.8B-Q4_K_M",
    "translation_model_path": "./models/hy-mt2",
    ```
  - `save()` 签名新增 4 个 keyword 参数:
    ```python
    def save(device=None, ..., translate_enabled=None, target_lang=None,
             translation_model=None, translation_model_path=None):
    ```
    每个参数 `is not None` 时更新 `current` 字典，与现有写法一致
  - 用户操作 → 立即持久化（类比现有点击 `device_radio` 就 `save(device=...)`）
    - `target_lang_dropdown.change()` → `save(target_lang=...)`
    - `translation_model_dropdown.change()` → `save(translation_model=...)`
    - `translate_enabled_checkbox.change()` → `save(translate_enabled=...)`
  - WebUI 启动时 `load()` 恢复上次选择，实现"记住用户最后一次选择"

- [ ] **Task 8: CUDA 兼容**
  - `src/cuda_setup.py` — 确认无需改动（llama-cpp-python 自带 CUDA 运行时查找）
  - `CLAUDE.md` — 补充 Windows 下 llama-cpp-python CUDA 安装:
    ```powershell
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    pip install llama-cpp-python
    ```
  - 推理时自动检测 CUDA → `n_gpu_layers=-1`；CPU → `n_gpu_layers=0`

- [ ] **Task 9: 测试**
  - `tests/test_translator.py` — 翻译器单元测试
    - `test_translator_load_unload` — 加载/卸载
    - `test_translate_single_segment` — 单段翻译
    - `test_translate_segments_preserves_timestamps` — 时间戳不丢失
    - `test_translate_empty_segments` — 空列表
    - `test_translate_stream_generator` — 流式逐段产出
    - `test_raises_when_model_missing` — 模型不存在抛异常
    - `test_translate_with_different_target_langs` — 多目标语言
  - `tests/test_translation_formatter.py` — 格式化测试
    - `test_bilingual_txt_format`
    - `test_translated_txt_format`
    - `test_translated_srt_format`
    - `test_bilingual_srt_format`
    - `test_write_with_translation_outputs_all_files`
    - `test_write_without_translation_unchanged`
  - `tests/test_translation_model_manager.py` — 模型管理测试
    - `test_list_models_empty_dir`
    - `test_list_models_detects_downloaded`
    - `test_get_model_path`
  - `tests/test_regression.py` — CLI 回归
    - `test_translate_flag_disabled_by_default` — 不带 --translate 时行为不变
    - `test_translate_pipeline_full` — `--translate -t zh` 全链路 (`@pytest.mark.slow`)
  - `tests/test_webui_regression.py` — WebUI 回归
    - `test_translate_workflow` — 勾选翻译 + 选择目标语言 + 流式双语预览 (`@pytest.mark.slow`)
    - `test_translate_disabled_default` — 默认翻译关闭，不加载 llama.cpp

- [ ] **Task 10: 文档更新**
  - `README.md` — 翻译功能章节：
    - 支持的模型列表 + 体积
    - CLI 示例: `python main.py <url> --translate -t zh`
    - WebUI 截图说明
  - `CLAUDE.md`:
    - 架构图更新：双引擎（CTranslate2 + llama.cpp）
    - 新增依赖 `llama-cpp-python` + CUDA 安装说明
    - 翻译模型下载命令
    - 推理参数说明（temperature/top_p/top_k/repetition_penalty）

---

### 关键设计决策

| 决策 | 选项 | 选型 | 理由 |
|------|------|------|------|
| 翻译粒度 | 逐段 / 全文拼接 | **逐段** | 保留时间戳对齐，流式预览友好 |
| 预览方式 | 双框(原文\|译文) / 单框双语 | **单框双语** | 用户说"同步显示原文和译文"，一个框里交替最直观 |
| 模型选择 | 固定一个 / 6选1 | **6选1，默认 Q4_K_M** | 用户说要可选，覆盖轻量到顶配 |
| 输出格式 | 纯译文 / 原文+译文+双语 | **全部输出** | 原文TXT/SRT保持，额外加译文TXT/SRT+双语TXT |
| 模型加载 | 启动时 / 按需 lazy | **按需 lazy** | 不翻译就不占内存，启动秒开 |
| Whisper→翻译传参 | 自动 / 手动指定源语言 | **自动** — Whisper 检测到的语言作为源语言 | 减少用户操作 |
| CUDA 层数 | 固定 / 自适应 | **自适应**: CUDA→ -1, CPU→ 0 | 一键切换设备 |
