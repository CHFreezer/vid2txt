"""Configuration constants for vid2txt."""

# Audio processing
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"

# Whisper model
DEFAULT_MODEL = "base"
SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large-v3")

# yt-dlp download options for audio-only
YT_DLP_AUDIO_FORMAT = "bestaudio/best"

# Required files for a complete model directory
# "vocabulary.*" — some models ship vocabulary.txt, others vocabulary.json
REQUIRED_MODEL_FILES = ("model.bin", "config.json", "tokenizer.json")

# Output
OUTPUT_ENCODING = "utf-8"
MAX_BASENAME_LENGTH = 100

# ---------------------------------------------------------------------------
# Translation (Hy-MT2 via llama-cpp-python)
# ---------------------------------------------------------------------------

# Target languages supported by Hy-MT2 (33 languages, short codes from README)
TARGET_LANGUAGE_CHOICES: list[tuple[str, str]] = [
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

# Available GGUF translation models (repo, filename, size)
TRANSLATION_MODEL_REPOS = {
    "1.8B-1.25Bit": {"repo": "tencent/Hy-MT2-1.8B-1.25Bit-GGUF", "filename": "Hy-MT2-1.8B-1.25Bit.gguf", "size_gb": 0.45},
    "1.8B-2Bit":    {"repo": "tencent/Hy-MT2-1.8B-2bit-GGUF",   "filename": "Hy-MT2-1.8B-2Bit.gguf",    "size_gb": 0.59},
    "1.8B-Q4_K_M":  {"repo": "tencent/Hy-MT2-1.8B-GGUF", "filename": "Hy-MT2-1.8B-Q4_K_M.gguf", "size_gb": 1.13},
    "1.8B-Q6_K":    {"repo": "tencent/Hy-MT2-1.8B-GGUF", "filename": "Hy-MT2-1.8B-Q6_K.gguf",   "size_gb": 1.47},
    "1.8B-Q8_0":    {"repo": "tencent/Hy-MT2-1.8B-GGUF", "filename": "Hy-MT2-1.8B-Q8_0.gguf",   "size_gb": 1.91},
    "7B-Q4_K_M":    {"repo": "tencent/Hy-MT2-7B-GGUF",   "filename": "Hy-MT2-7B-Q4_K_M.gguf",   "size_gb": 4.62},
    "7B-Q6_K":      {"repo": "tencent/Hy-MT2-7B-GGUF",   "filename": "HY-MT2-7B-Q6_K.gguf",     "size_gb": 6.16},
    "7B-Q8_0":      {"repo": "tencent/Hy-MT2-7B-GGUF",   "filename": "HY-MT2-7B-Q8_0.gguf",     "size_gb": 7.98},
}

SUPPORTED_TRANSLATION_MODELS = tuple(TRANSLATION_MODEL_REPOS.keys())

DEFAULT_TRANSLATION_MODEL = "1.8B-1.25Bit"
DEFAULT_TRANSLATION_MODEL_DIR = "./models/hy-mt2"

# Inference hyper-params (from Hy-MT2 official README)
TRANSLATION_INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.6,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "max_tokens": 4096,
}
