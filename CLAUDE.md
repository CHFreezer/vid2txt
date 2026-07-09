# CLAUDE.md

vid2txt — 视频语音转文字（Bilibili / YouTube），faster-whisper + ctranslate2.

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

`src/vid2txt/cli.py` — 命令行流水线（下载→转录→格式化）
`src/vid2txt/webui.py` — Gradio 界面
`src/vid2txt/cuda_setup.py` — CUDA DLL 预加载（Windows / Linux）
`src/vid2txt/model_manager.py` — Whisper 模型发现 & 下载
`src/vid2txt/config.py` — 模型尺寸、采样率等常量

## 模型下载

- 模型存储路径：`./models/faster-whisper-{size}/`
- 下载时禁用 Xet 存储（`HF_HUB_DISABLE_XET`），否则进度条不更新
- 逐文件 `hf_hub_download`（非 `snapshot_download`），保证进度条准确

## WebUI 进度条

- 下载 / 转录的进度条通过 `show_progress_on=progress_area` 锚定到页面底部
- `status_md` 组件显示状态文字，`progress_area` 组件承载 Gradio 进度条
- 不使用自定义 HTML 进度条，全用 Gradio 原生组件
