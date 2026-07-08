# vid2txt

Bilibili 视频语音转文字，基于 Whisper (faster-whisper + ctranslate2)。

## 快速开始

```bash
# 激活 conda 环境
conda activate ./.conda

# 命令行
python main.py https://www.bilibili.com/video/BV1GJ41177UQ
python main.py https://b23.tv/xxxxxx -o ./output -m large-v3 --language zh -v

# WebUI（或双击 run.bat）
python webui.py
# → http://127.0.0.1:7860
```

## 项目结构

```
vid2txt/
├── main.py              # CLI 入口
├── webui.py             # Gradio WebUI 入口
├── run.bat              # 双击启动 WebUI（Windows 专用）
├── requirements.txt
├── src/vid2txt/
│   ├── cli.py           # 命令行参数 & 流水线
│   ├── webui.py         # WebUI 界面
│   ├── downloader.py    # yt-dlp 下载音频
│   ├── transcriber.py   # faster-whisper 转录
│   ├── formatter.py     # TXT + SRT 输出
│   ├── cuda_setup.py    # Windows CUDA DLL 预加载
│   ├── config.py        # 模型、采样率等常量
│   └── utils.py         # URL 校验、依赖检查等
└── .conda/              # Conda 环境（git-ignored）
```

## 环境

- Python 3.12，conda 环境在 `.conda/`
- 依赖：`faster-whisper >= 1.1.0`、`gradio >= 5.0`
- 外部依赖：`ffmpeg`（音频转换）
- CUDA：通过 pip 安装的 `nvidia-cuda-runtime-cu12`、`nvidia-cublas-cu12`，Windows 上由 `cuda_setup.py` 预加载 DLL

## 模型

`tiny` | `base` | `small` | `medium`（默认） | `large-v3`
