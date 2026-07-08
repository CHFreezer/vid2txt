"""Configuration constants for vid2txt."""

# Audio processing
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"

# Whisper model
DEFAULT_MODEL = "medium"
SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large-v3")

# yt-dlp download options for audio-only
YT_DLP_AUDIO_FORMAT = "bestaudio/best"

# Output
OUTPUT_ENCODING = "utf-8"
MAX_BASENAME_LENGTH = 100
