"""
src/components/data_ingestion.py
---------------------------------
Stage 1: Data Ingestion
Downloads the raw EmpatheticDialogues dataset from Hugging Face Hub
and saves each split (train/validation/test) as raw CSV files under
artifacts/data_ingestion/. This stage does NOT clean or restructure
the data -- that's the job of data_transformation.py.
"""

import os
import sys
import pandas as pd
from datasets import load_dataset

from src.logger import logging
from src.exception import ChatbotException
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def download_dataset(self):
        try:
            logging.info(f"Downloading dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name)
            return dataset
        except Exception as e:
            raise ChatbotException(e, sys)

    def save_raw_splits(self, dataset) -> dict:
        try:
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            raw_paths = {}

            for split in ["train", "validation", "test"]:
                df = pd.DataFrame(dataset[split])
                out_path = os.path.join(self.config.raw_data_dir, f"{split}_raw.csv")
                df.to_csv(out_path, index=False)
                raw_paths[split] = out_path
                logging.info(f"Saved raw split '{split}' ({len(df)} rows) -> {out_path}")

            return raw_paths
        except Exception as e:
            raise ChatbotException(e, sys)

    def initiate_data_ingestion(self) -> dict:
        logging.info("===== Data Ingestion started =====")
        dataset = self.download_dataset()
        raw_paths = self.save_raw_splits(dataset)
        logging.info("===== Data Ingestion completed =====")
        return raw_paths


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
