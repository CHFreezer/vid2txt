"""Configuration constants for vid2txt."""

# Audio processing
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"

# Whisper model
DEFAULT_MODEL = "base"
SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_WHISPER_MODEL_DIR = "./models/faster-whisper"

# Default language settings
DEFAULT_LANGUAGE = "auto"
DEFAULT_TARGET_LANG = "zh"

# yt-dlp download options for audio-only
YT_DLP_AUDIO_FORMAT = "bestaudio/best"

# Required files for a complete model directory
# "vocabulary.*" — some models ship vocabulary.txt, others vocabulary.json
REQUIRED_MODEL_FILES = ("model.bin", "config.json", "tokenizer.json")

# Output
OUTPUT_ENCODING = "utf-8"
MAX_BASENAME_LENGTH = 100

# ---------------------------------------------------------------------------
# Translation (M2M100 via CTranslate2)
# ---------------------------------------------------------------------------

# Target languages (same as LANGUAGE_CHOICES minus "auto")
TARGET_LANGUAGE_CHOICES: list[tuple[str, str]] = [
    ("中文", "zh"), ("English", "en"),
    ("日本語", "ja"), ("한국어", "ko"),
]

# Translation model (CTranslate2-converted M2M100)
TRANSLATION_MODEL_REPO = "gn64/M2M100_418M_CTranslate2"
TRANSLATION_MODEL_SIZE_GB = 0.50  # int8 quantized
DEFAULT_TRANSLATION_MODEL_DIR = "./models/m2m100"

# Tokenizer source (separate from CTranslate2 model)
TRANSLATION_TOKENIZER_REPO = "facebook/m2m100_418M"

# Required files for a complete translation model directory
REQUIRED_TRANSLATION_MODEL_FILES = (
    "model.bin", "config.json", "shared_vocabulary.json",
)
