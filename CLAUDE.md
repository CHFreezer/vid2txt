# CLAUDE.md

vid2txt — Bilibili 视频语音转文字，faster-whisper + ctranslate2.

## 环境

Python 3.12+，conda / venv / pip 均可。依赖见 `requirements.txt`。

## 入口

- **CLI**: `python main.py <bilibili_url> [options]`
- **WebUI**: `python webui.py` → http://127.0.0.1:7860

## 导入顺序（重要）

`cuda_setup.py` 必须在任何 CUDA 相关 import 之前调用。`main.py` 和 `webui.py` 顶部都已处理。

## 外部依赖

`ffmpeg` 用于音频转换，运行时由 `check_dependencies()` 检查。

## 关键文件

`src/vid2txt/cli.py` — 命令行流水线（下载→转录→格式化）
`src/vid2txt/webui.py` — Gradio 界面
`src/vid2txt/cuda_setup.py` — Windows CUDA DLL 预加载
`src/vid2txt/config.py` — 模型尺寸、采样率等常量
