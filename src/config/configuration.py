"""
src/config/configuration.py
-----------------------------
ConfigurationManager: reads config.yaml + params.yaml and builds
the typed config-entity dataclasses each component expects.

This is the standard "config -> entity" pattern used in production
ML repos: YAML files are human-editable, but components only ever
deal with typed Python objects (no raw dict key lookups scattered
around the codebase).
"""

import sys
from src.logger import logging
from src.exception import ChatbotException
from src.utils import read_yaml
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)
from src.constants import SYSTEM_PROMPT

CONFIG_FILE_PATH = "config/config.yaml"
PARAMS_FILE_PATH = "params.yaml"


class ConfigurationManager:
    def __init__(self, config_path: str = CONFIG_FILE_PATH, params_path: str = PARAMS_FILE_PATH):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
        except Exception as e:
            raise ChatbotException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        c = self.config["data_ingestion"]
        return DataIngestionConfig(
            dataset_name=c["dataset_name"],
            raw_data_dir=c["raw_data_dir"],
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        c = self.config["data_transformation"]
        return DataTransformationConfig(
            transformed_data_dir=c["transformed_data_dir"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        c = self.config["model_trainer"]
        p = self.params["training_params"]
        return ModelTrainerConfig(
            base_model_name=c["base_model_name"],
            model_output_dir=c["model_output_dir"],
            max_seq_length=c["max_seq_length"],
            max_train_examples=p["max_train_examples"],
            max_eval_examples=p["max_eval_examples"],
            num_train_epochs=p["num_train_epochs"],
            train_batch_size=p["train_batch_size"],
            eval_batch_size=p["eval_batch_size"],
            gradient_accumulation_steps=p["gradient_accumulation_steps"],
            learning_rate=float(p["learning_rate"]),
            weight_decay=p["weight_decay"],
            warmup_steps=p["warmup_steps"],
            system_prompt=SYSTEM_PROMPT,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        c = self.config["model_evaluation"]
        mlflow_c = self.config["mlflow"]
        return ModelEvaluationConfig(
            evaluation_report_path=c["evaluation_report_path"],
            num_eval_samples=c["num_eval_samples"],
            mlflow_tracking_uri=mlflow_c["tracking_uri"],
            mlflow_experiment_name=mlflow_c["experiment_name"],
        )

    def get_prediction_config(self) -> PredictionConfig:
        c = self.config["model_trainer"]
        return PredictionConfig(
            model_path=c["model_output_dir"],
            system_prompt=SYSTEM_PROMPT,
        )
