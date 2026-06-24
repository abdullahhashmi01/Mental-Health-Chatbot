"""
tests/test_config.py
------------------------
Unit tests for configuration loading -- ensures config.yaml and
params.yaml are valid and produce correctly-typed config objects.

Run:
    pytest tests/test_config.py -v
"""

from src.config.configuration import ConfigurationManager
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


def test_configuration_manager_loads_yaml_files():
    config_manager = ConfigurationManager()
    assert config_manager.config is not None
    assert config_manager.params is not None


def test_data_ingestion_config():
    config_manager = ConfigurationManager()
    cfg = config_manager.get_data_ingestion_config()
    assert isinstance(cfg, DataIngestionConfig)
    assert cfg.dataset_name == "empathetic_dialogues"


def test_data_transformation_config():
    config_manager = ConfigurationManager()
    cfg = config_manager.get_data_transformation_config()
    assert isinstance(cfg, DataTransformationConfig)
    assert "data_transformation" in cfg.transformed_data_dir


def test_model_trainer_config_has_valid_hyperparams():
    config_manager = ConfigurationManager()
    cfg = config_manager.get_model_trainer_config()
    assert isinstance(cfg, ModelTrainerConfig)
    assert cfg.num_train_epochs > 0
    assert cfg.train_batch_size > 0
    assert 0 < cfg.learning_rate < 1


def test_model_evaluation_config():
    config_manager = ConfigurationManager()
    cfg = config_manager.get_model_evaluation_config()
    assert isinstance(cfg, ModelEvaluationConfig)
    assert cfg.num_eval_samples > 0
