"""
src/entity/config_entity.py
----------------------------
Dataclasses that hold configuration for each pipeline stage.
Keeping config as typed objects (instead of passing raw dicts/strings
around) is a common industry pattern -- it makes each component's
inputs explicit and easy to test.
"""

from dataclasses import dataclass
from src import constants as c


@dataclass
class DataIngestionConfig:
    dataset_name: str = c.DATASET_NAME
    raw_data_dir: str = c.RAW_DATA_DIR


@dataclass
class DataTransformationConfig:
    transformed_data_dir: str = c.TRANSFORMED_DATA_DIR


@dataclass
class ModelTrainerConfig:
    base_model_name: str = c.BASE_MODEL_NAME
    model_output_dir: str = c.MODEL_OUTPUT_DIR
    max_seq_length: int = c.MAX_SEQ_LENGTH
    max_train_examples: int = c.MAX_TRAIN_EXAMPLES
    max_eval_examples: int = c.MAX_EVAL_EXAMPLES
    num_train_epochs: int = c.NUM_TRAIN_EPOCHS
    train_batch_size: int = c.TRAIN_BATCH_SIZE
    eval_batch_size: int = c.EVAL_BATCH_SIZE
    gradient_accumulation_steps: int = c.GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = c.LEARNING_RATE
    weight_decay: float = c.WEIGHT_DECAY
    warmup_steps: int = c.WARMUP_STEPS
    system_prompt: str = c.SYSTEM_PROMPT


@dataclass
class PredictionConfig:
    model_path: str = c.MODEL_OUTPUT_DIR
    system_prompt: str = c.SYSTEM_PROMPT
