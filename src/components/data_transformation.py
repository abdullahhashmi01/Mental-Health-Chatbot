"""
src/components/data_transformation.py
----------------------------------------
Stage 2: Data Transformation
Reads the raw CSVs produced by data_ingestion.py, groups utterances
by conversation, and converts them into clean (context, response)
training pairs. Also handles text cleaning and deduplication.
"""

import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import ChatbotException
from src.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    @staticmethod
    def _clean_text(text: str) -> str:
        return str(text).replace("_comma_", ",").strip()

    def _build_pairs_from_raw(self, raw_csv_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(raw_csv_path)
            pairs = []

            for conv_id, group in df.groupby("conv_id"):
                turns = group.sort_values("utterance_idx")
                turns = turns.to_dict("records")

                for i in range(len(turns) - 1):
                    context = self._clean_text(turns[i]["utterance"])
                    response = self._clean_text(turns[i + 1]["utterance"])
                    emotion = turns[i].get("context", "")
                    pairs.append({
                        "emotion": emotion,
                        "context": context,
                        "response": response,
                    })

            pairs_df = pd.DataFrame(pairs)
            pairs_df = pairs_df[
                (pairs_df["context"].str.len() > 2) & (pairs_df["response"].str.len() > 2)
            ]
            pairs_df.drop_duplicates(subset=["context", "response"], inplace=True)
            return pairs_df
        except Exception as e:
            raise ChatbotException(e, sys)

    def initiate_data_transformation(self, raw_paths: dict) -> dict:
        try:
            logging.info("===== Data Transformation started =====")
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)
            transformed_paths = {}

            for split, raw_path in raw_paths.items():
                logging.info(f"Transforming split: {split}")
                pairs_df = self._build_pairs_from_raw(raw_path)
                out_path = os.path.join(self.config.transformed_data_dir, f"{split}.csv")
                pairs_df.to_csv(out_path, index=False)
                transformed_paths[split] = out_path
                logging.info(f"Saved {len(pairs_df)} clean pairs -> {out_path}")

            logging.info("===== Data Transformation completed =====")
            return transformed_paths
        except Exception as e:
            raise ChatbotException(e, sys)


if __name__ == "__main__":
    # Allows running this stage independently if raw data already exists
    from src.constants import RAW_DATA_DIR
    raw = {
        "train": os.path.join(RAW_DATA_DIR, "train_raw.csv"),
        "validation": os.path.join(RAW_DATA_DIR, "validation_raw.csv"),
        "test": os.path.join(RAW_DATA_DIR, "test_raw.csv"),
    }
    DataTransformation().initiate_data_transformation(raw)
