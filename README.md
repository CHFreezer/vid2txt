# vid2txt

视频语音转文字（Bilibili / YouTube / YouTube Shorts），基于 Whisper (faster-whisper + ctranslate2)。

## 快速开始

```bash
pip install -r requirements.txt

# 命令行
python main.py https://www.bilibili.com/video/BV1GJ41177UQ
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -o ./output -m base --language ja
python main.py https://b23.tv/xxxxxx -o ./output -m large-v3 --language zh -v

# WebUI
python webui.py
# → http://127.0.0.1:7860
```

## 项目结构

```
vid2txt/
├── main.py              # CLI 入口
├── webui.py             # Gradio WebUI 入口
├── requirements.txt
├── src/vid2txt/
│   ├── cli.py           # 命令行参数 & 流水线
│   ├── webui.py         # WebUI 界面
│   ├── downloader.py    # yt-dlp 下载音频
│   ├── transcriber.py   # faster-whisper 转录
│   ├── formatter.py     # TXT + SRT 输出
│   ├── model_manager.py # 模型发现 & 下载
│   ├── cuda_setup.py    # CUDA DLL 预加载（Windows / Linux）
│   ├── config.py        # 模型、采样率等常量
│   ├── utils.py         # URL 校验、依赖检查等
│   └── settings.py      # 用户设置持久化
├── models/              # Whisper 模型（git-ignored）
├── temp/                # 临时音频（git-ignored）
├── output/              # 转录输出（git-ignored）
└── webui_outputs/       # WebUI 输出（git-ignored）
```

## 环境

- Python 3.12+，任意包管理器（conda / venv / pip）均可
- 依赖：`faster-whisper >= 1.1.0`、`gradio >= 5.0`
- 外部工具：`ffmpeg`（音频转换）、`yt-dlp`（视频下载）
  - `yt-dlp` 需保持最新（视频网站接口频繁变化），推荐通过 `choco install yt-dlp` / `choco upgrade yt-dlp` 管理
- **Windows / Linux CUDA**：pip 安装 `nvidia-cuda-runtime-cu12`、`nvidia-cublas-cu12`，由 `cuda_setup.py` 预加载 DLL / SO
- **macOS**：faster-whisper 不支持 GPU 加速，自动回退 CPU（int8 + ARM NEON）。需 GPU 加速可换用 whisper.cpp + CoreML 或 MLX-Whisper

## 模型

`tiny` | `base`（默认） | `small` | `medium` | `large-v3`

WebUI 支持一键下载模型到 `./models/` 目录，首次使用需下载（150MB~3.5GB）。
