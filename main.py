"""
main.py
-------
Single entry point to run the full training pipeline:
    Data Ingestion -> Data Transformation -> Model Training

Run:
    python main.py
"""

from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    model_path = pipeline.run_pipeline()
    print(f"\nTraining complete. Fine-tuned model saved at: {model_path}")
