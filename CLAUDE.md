# CLAUDE.md

vid2txt — 视频语音转文字（Bilibili / YouTube / Shorts），faster-whisper + ctranslate2.

## 环境

Python 3.12+，conda / venv / pip 均可。依赖见 `requirements.txt`。

## 入口

- **CLI**: `python main.py <bilibili_url> [options]`
- **WebUI**: `python webui.py` → http://127.0.0.1:7860`

## 导入顺序（重要）

`cuda_setup.py` 必须在任何 CUDA 相关 import 之前调用。`main.py` 和 `webui.py` 顶部都已处理。

## 运行时依赖

- `ffmpeg` — 音频转换
- `yt-dlp` — 视频下载，需保持最新（视频网站接口频繁变动）。不在 requirements.txt 中管理，由用户自行通过 `pip install -U yt-dlp` / `choco upgrade yt-dlp` 更新
- 两者由 `check_dependencies()` 检查是否存在，缺失时给出安装提示

## 关键文件

`src/cli.py` — 命令行流水线（下载→转录→格式化）
`src/webui.py` — Gradio 界面
`src/cuda_setup.py` — CUDA DLL 预加载（Windows / Linux）
`src/model_manager.py` — Whisper 模型发现 & 下载
`src/config.py` — 模型尺寸、采样率等常量

## 模型下载

- 模型存储路径：`./models/faster-whisper-{size}/`
- 下载时禁用 Xet 存储（`HF_HUB_DISABLE_XET`），否则进度条不更新
- 逐文件 `hf_hub_download`（非 `snapshot_download`），保证进度条准确

## WebUI 进度条

- 下载 / 转录的进度条通过 `show_progress_on=progress_area` 锚定到页面底部
- `status_md` 组件显示状态文字，`progress_area` 组件承载 Gradio 进度条
- 不使用自定义 HTML 进度条，全用 Gradio 原生组件

## 测试

### 运行

```bash
# 必须在 conda 环境内运行（pytest / playwright 安装在 .conda 中）
# 激活命令（pwsh，遵循 memory/windows-encoding 约定）：
pwsh -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; . C:\Users\chfre\miniconda3\shell\condabin\conda-hook.ps1; conda activate 'E:\CHWork\Python Projects\vid2txt\.conda'; <命令>"

# 快速（跳过所有下载+转录，~45s）
python -m pytest tests/ -m "not slow"

# 完整（含 CLI + WebUI 全链路，~3min）
python -m pytest tests/

# 单项
python -m pytest tests/test_utils.py -v               # 单元测试
python -m pytest tests/test_regression.py -v           # CLI 回归
python -m pytest tests/test_webui_regression.py -v     # WebUI 回归
```

### 分层策略

| 文件 | 类型 | 覆盖 |
|------|------|------|
| `tests/test_utils.py` | 单元 | `validate_url`, `format_timestamp`, `get_output_basename`, `check_dependencies` — 纯函数，无副作用 |
| `tests/test_formatter.py` | 单元 | `Formatter.to_txt`, `Formatter.to_srt` — 空片段、单/多片段、SRT 序号递增 |
| `tests/test_regression.py` | CLI 回归 | 三个固定视频 URL 验证 + yt-dlp 元数据获取 + YouTube Shorts 完整链路（下载→转换→转录 tiny→格式化），含 stream vs batch API 对等验证 |
| `tests/test_webui_regression.py` | WebUI 回归 | Playwright + Chromium，同进程线程启动 Gradio，Smoke → Analyse（三个 URL）→ Settings → 完整转录链路 → Stop 按钮 |

### 固定测试视频

```
Bilibili:      https://www.bilibili.com/video/BV1Kt7q6hEAr
YouTube:       https://www.youtube.com/watch?v=dQw4w9WgXcQ
YouTube Shorts: https://www.youtube.com/shorts/Lp1o_IDZ7vk
```

### Marker

- `@pytest.mark.slow` — 需要下载音频 + 模型推理的测试，CI 快速检查时跳过
- 配置在 `pyproject.toml` 的 `[tool.pytest.ini_options]`

### 注意事项

- WebUI 测试使用 **7860 端口**（与 `webui.py` 硬编码一致），启动前会自动 kill 占用进程
- WebUI 服务器在同进程 daemon 线程中运行（`_build_ui()` + `demo.launch(inbrowser=False)`），测试结束自动 `demo.close()`
- CLI 回归测试使用 CPU + int8（`device="cpu", compute_type="int8"`），不依赖 CUDA
- 转录测试使用 `tiny` 模型，首次运行会下载到 HF cache（~150MB）
