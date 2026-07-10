# vid2txt

视频语音转文字（Bilibili / YouTube / YouTube Shorts），基于 Whisper (faster-whisper + ctranslate2)。
可选翻译功能：M2M100-418M + CTranslate2，100 种语言。

## 快速开始

```bash
pip install -r requirements.txt

# 命令行（转录）
python main.py https://www.bilibili.com/video/BVxxxxxxxxx
python main.py https://www.youtube.com/watch?v=xxxxxxxxxxx -o ./output -m base --language ja

# 命令行（转录 + 翻译）
python main.py <url> --translate -t en

# WebUI
python webui.py
# → http://127.0.0.1:7860
```

## 翻译功能（可选）

基于 Facebook [M2M100-418M](https://huggingface.co/facebook/m2m100_418M) + CTranslate2 int8，默认关闭。

### CLI

```bash
python main.py <url> --translate -t en   # 翻译为英文
python main.py <url> --translate -t ja   # 翻译为日文
```

### WebUI

在"模型与设备设置"面板中勾选"启用翻译"，选择目标语言，点击下载模型后即可使用。

### 依赖

翻译功能通过 `transformers`（仅 tokenizer）+ `sentencepiece` 实现，无需额外推理引擎。

### 模型

M2M100-418M int8（`gn64/M2M100_418M_CTranslate2`），约 468MB，支持 100 种语言。存储于 `./models/m2m100/`。

## 项目结构

```
vid2txt/
├── main.py              # CLI 入口
├── webui.py             # Gradio WebUI 入口
├── requirements.txt
├── src/
│   ├── cli.py           # 命令行参数 & 流水线
│   ├── webui.py         # WebUI 界面
│   ├── downloader.py    # yt-dlp 下载音频
│   ├── transcriber.py   # faster-whisper 转录
│   ├── translator.py    # M2M100 CTranslate2 翻译
│   ├── formatter.py     # TXT + SRT + 双语输出
│   ├── model_manager.py # Whisper 模型发现 & 下载
│   ├── translation_model_manager.py # M2M100 模型发现 & 下载
│   ├── cuda_setup.py    # CUDA DLL 预加载（Windows / Linux）
│   ├── config.py        # 模型尺寸、语言、推理参数等常量
│   ├── utils.py         # URL 校验、依赖检查等
│   └── settings.py      # 用户设置持久化
├── models/              # 模型存储（git-ignored）
│   ├── faster-whisper/  # Whisper 模型
│   └── m2m100/          # 翻译模型
├── temp/                # 临时音频（git-ignored）
├── output/              # CLI 输出（git-ignored）
└── webui_outputs/       # WebUI 输出（git-ignored）
```

## 环境

- Python 3.12+，任意包管理器（conda / venv / pip）均可
- 依赖：`faster-whisper >= 1.1.0`、`gradio >= 5.0`、`transformers >= 4.45`
- 外部工具：`ffmpeg`（音频转换）、`yt-dlp`（视频下载）
  - `yt-dlp` 需保持最新（视频网站接口频繁变化），推荐通过 `choco install yt-dlp` / `choco upgrade yt-dlp` 管理
- **Windows / Linux CUDA**：pip 安装 `nvidia-cuda-runtime-cu12`、`nvidia-cublas-cu12`，由 `cuda_setup.py` 预加载 DLL / SO
- **macOS**：faster-whisper 不支持 GPU 加速，自动回退 CPU（int8 + ARM NEON）。需 GPU 加速可换用 whisper.cpp + CoreML 或 MLX-Whisper

## 模型

`tiny` | `base`（默认） | `small` | `medium` | `large-v3`

WebUI 支持一键下载模型到 `./models/faster-whisper/` 目录，首次使用需下载（150MB~3.5GB）。

## 测试

```bash
pip install pytest pytest-playwright

# 快速测试（跳过下载+转录，~45s）
pytest tests/ -m "not slow"

# 完整回归（含 CLI + WebUI 全链路，~3min）
pytest tests/
```

| 层级 | 命令 | 覆盖 |
|------|------|------|
| 单元测试 | `pytest tests/test_utils.py tests/test_formatter.py` | URL 校验、时间戳、TXT/SRT/双语格式化 |
| 翻译模型 | `pytest tests/test_translation_model_manager.py` | 模型发现、路径解析 |
| CLI 回归 | `pytest tests/test_regression.py` | 三个固定视频的元数据 + 完整转录链路 |
| WebUI 回归 | `pytest tests/test_webui_regression.py` | Playwright 浏览器自动化，Smoke → Analyse → Transcribe → Stop |
