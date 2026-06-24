"""
src/constants.py
-----------------
Single source of truth for fixed values used across components.
"""

# ---- Dataset ----
DATASET_NAME = "empathetic_dialogues"
RAW_DATA_DIR = "artifacts/data_ingestion"
TRANSFORMED_DATA_DIR = "artifacts/data_transformation"

# ---- Model ----
BASE_MODEL_NAME = "distilgpt2"
MODEL_OUTPUT_DIR = "artifacts/model_trainer/distilgpt2-empathetic"
MAX_SEQ_LENGTH = 128

# ---- Training (CPU-friendly defaults) ----
MAX_TRAIN_EXAMPLES = 8000
MAX_EVAL_EXAMPLES = 500
NUM_TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

# ---- Persona / tone ----
SYSTEM_PROMPT = (
    "You are a kind, supportive, and emotionally gentle assistant. "
    "You listen carefully and respond with warmth, validation, and care."
)

# ---- Safety ----
CRISIS_KEYWORDS = [
    "kill myself", "suicide", "end my life", "want to die",
    "hurting myself", "self harm", "self-harm", "cut myself",
    "no reason to live", "can't go on", "better off dead",
]

CRISIS_RESPONSE = (
    "I'm really glad you told me how you're feeling, and I'm concerned about you. "
    "You deserve support from someone who can really help right now. "
    "If you are in immediate danger, please contact your local emergency number. "
    "You can also reach a crisis line in your country -- for example, in the US you can "
    "call or text 988 (Suicide & Crisis Lifeline), and internationally you can find a helpline "
    "at https://findahelpline.com. "
    "Please consider reaching out to someone you trust or a mental health professional. "
    "I'm here to listen too, if you'd like to keep talking."
)
