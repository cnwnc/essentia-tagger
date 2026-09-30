"""Shared constants for the tagger pipeline (the only place schema details live).

The track-JSON wire format (Contract A in ARCHITECTURE.md) is FROZEN:
{
  "schema": 1, "path": str, "artist": str, "album": str, "duration_sec": float,
  "models": {embedding, embedding_layer, head, head_output},
  "embedding": {dtype: "float32", shape: [768], layer, data: <base64>},
  "activations": {"Parent---Style": float, ...}   # 519 keys, human-readable
}
"""

SCHEMA_VERSION = 1
SAMPLE_RATE = 16000
N_CLASSES = 519
EMB_DIM = 768
# MAEST's 30s window (1876 mel frames) errors below ~30.02s of audio;
# tracks shorter than this are padded with silence at load time.
MIN_SECONDS = 30.5
AUDIO_EXTS = {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.opus', '.wma'}

# Model files come from the nix store via env (flake.nix exports these).
ENV_EMBEDDING_MODEL = 'ESSENTIA_EMBEDDING_MODEL'
ENV_CLASSIFIER_HEAD = 'ESSENTIA_CLASSIFIER_HEAD'
ENV_CLASSES_JSON = 'ESSENTIA_CLASSES_JSON'
