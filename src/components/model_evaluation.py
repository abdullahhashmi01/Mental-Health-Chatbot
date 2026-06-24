"""
src/components/model_evaluation.py
--------------------------------------
Stage 4: Model Evaluation
Generates replies for a sample of the test set, scores them against
the real reference responses using BLEU (sacrebleu) and ROUGE
(rouge_score), and logs everything to MLflow for experiment tracking.
"""

import os
import sys
import json
import pandas as pd
import mlflow
import sacrebleu
from rouge_score import rouge_scorer

from src.logger import logging
from src.exception import ChatbotException
from src.entity.config_entity import ModelEvaluationConfig
from src.pipeline.prediction_pipeline import PredictionPipeline


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig = ModelEvaluationConfig()):
        self.config = config
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def _score_pair(self, prediction: str, reference: str) -> dict:
        bleu = sacrebleu.sentence_bleu(prediction, [reference]).score
        rouge_scores = self.rouge.score(reference, prediction)
        return {
            "bleu": bleu,
            "rouge1_f": rouge_scores["rouge1"].fmeasure,
            "rougeL_f": rouge_scores["rougeL"].fmeasure,
        }

    def initiate_model_evaluation(self, test_csv_path: str) -> dict:
        try:
            logging.info("===== Model Evaluation started =====")

            df = pd.read_csv(test_csv_path).dropna(subset=["context", "response"])
            df = df.sample(n=min(self.config.num_eval_samples, len(df)), random_state=42)

            pipeline = PredictionPipeline()

            all_scores = []
            for row in df.itertuples():
                prediction = pipeline.generate_reply(row.context)
                scores = self._score_pair(prediction, row.response)
                all_scores.append(scores)

            avg_bleu = sum(s["bleu"] for s in all_scores) / len(all_scores)
            avg_rouge1 = sum(s["rouge1_f"] for s in all_scores) / len(all_scores)
            avg_rougeL = sum(s["rougeL_f"] for s in all_scores) / len(all_scores)

            metrics = {
                "num_samples": len(all_scores),
                "avg_bleu": round(avg_bleu, 4),
                "avg_rouge1_f": round(avg_rouge1, 4),
                "avg_rougeL_f": round(avg_rougeL, 4),
            }

            os.makedirs(os.path.dirname(self.config.evaluation_report_path), exist_ok=True)
            with open(self.config.evaluation_report_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logging.info(f"Evaluation metrics: {metrics}")

            # ---- MLflow logging ----
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            with mlflow.start_run():
                mlflow.log_param("num_eval_samples", metrics["num_samples"])
                mlflow.log_metric("avg_bleu", metrics["avg_bleu"])
                mlflow.log_metric("avg_rouge1_f", metrics["avg_rouge1_f"])
                mlflow.log_metric("avg_rougeL_f", metrics["avg_rougeL_f"])
                mlflow.log_artifact(self.config.evaluation_report_path)

            logging.info("===== Model Evaluation completed =====")
            return metrics
        except Exception as e:
            raise ChatbotException(e, sys)


if __name__ == "__main__":
    from src.constants import TRANSFORMED_DATA_DIR
    ModelEvaluation().initiate_model_evaluation(
        os.path.join(TRANSFORMED_DATA_DIR, "test.csv")
    )
