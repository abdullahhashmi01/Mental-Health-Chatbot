"""
src/pipeline/training_pipeline.py
------------------------------------
Orchestrates the full training pipeline:
    Data Ingestion -> Data Transformation -> Model Training

This is the single entry point used by main.py to run training
end-to-end.
"""

import sys
from src.logger import logging
from src.exception import ChatbotException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerComponent


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainerComponent()

    def run_pipeline(self):
        try:
            logging.info("########## TRAINING PIPELINE STARTED ##########")

            raw_paths = self.data_ingestion.initiate_data_ingestion()
            transformed_paths = self.data_transformation.initiate_data_transformation(raw_paths)
            model_path = self.model_trainer.initiate_model_training(transformed_paths)

            logging.info(f"########## TRAINING PIPELINE COMPLETED. Model at: {model_path} ##########")
            return model_path
        except Exception as e:
            raise ChatbotException(e, sys)


if __name__ == "__main__":
    TrainingPipeline().run_pipeline()
