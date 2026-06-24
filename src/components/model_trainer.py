"""
src/components/model_trainer.py
---------------------------------
Stage 3: Model Training
Fine-tunes the base causal LM (DistilGPT2 by default) on the
transformed (context, response) pairs using Hugging Face's
Trainer API. CPU-friendly settings by default.
"""

import os
import sys
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.logger import logging
from src.exception import ChatbotException
from src.entity.config_entity import ModelTrainerConfig


class ModelTrainerComponent:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config
        self.tokenizer = None

    def _load_split(self, path: str, limit: int) -> pd.DataFrame:
        df = pd.read_csv(path).dropna(subset=["context", "response"])
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=42)
        return df

    def _format_example(self, context: str, response: str) -> str:
        return (
            f"{self.config.system_prompt}\n\n"
            f"User: {context}\n"
            f"Assistant: {response}{self.tokenizer.eos_token}"
        )

    def _tokenize_dataset(self, df: pd.DataFrame) -> Dataset:
        texts = [self._format_example(r.context, r.response) for r in df.itertuples()]
        ds = Dataset.from_dict({"text": texts})

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        return ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    def initiate_model_training(self, transformed_paths: dict) -> str:
        try:
            logging.info("===== Model Training started =====")
            logging.info(f"Loading base model: {self.config.base_model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name)

            train_df = self._load_split(transformed_paths["train"], self.config.max_train_examples)
            val_df = self._load_split(transformed_paths["validation"], self.config.max_eval_examples)

            logging.info(f"Train examples: {len(train_df)} | Val examples: {len(val_df)}")

            train_dataset = self._tokenize_dataset(train_df)
            val_dataset = self._tokenize_dataset(val_df)

            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

            training_args = TrainingArguments(
                output_dir=self.config.model_output_dir,
                overwrite_output_dir=True,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                logging_steps=50,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                report_to="none",
                no_cuda=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )

            logging.info("Starting fine-tuning...")
            trainer.train()

            os.makedirs(self.config.model_output_dir, exist_ok=True)
            trainer.save_model(self.config.model_output_dir)
            self.tokenizer.save_pretrained(self.config.model_output_dir)

            logging.info(f"Model saved -> {self.config.model_output_dir}")
            logging.info("===== Model Training completed =====")
            return self.config.model_output_dir
        except Exception as e:
            raise ChatbotException(e, sys)


if __name__ == "__main__":
    from src.constants import TRANSFORMED_DATA_DIR
    paths = {
        "train": os.path.join(TRANSFORMED_DATA_DIR, "train.csv"),
        "validation": os.path.join(TRANSFORMED_DATA_DIR, "validation.csv"),
    }
    ModelTrainerComponent().initiate_model_training(paths)
