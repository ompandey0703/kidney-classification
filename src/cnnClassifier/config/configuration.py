import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier.entity.config_entity import TrainingConfig

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=config.source_url,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.prepare_base_model

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_model_path=config.updated_model_path,
            params_image_size=params.IMAGE_SIZE,
            params_weights=params.WEIGHTS,
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_classes=params.CLASSES
        )
        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.train_model
        prepare_base_model = self.config.prepare_base_model
        params = self.params.prepare_base_model
        training_data_path = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-CT_Scan/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")

        training_config = TrainingConfig(
            root_dir=Path(self.config.artifacts_root),
            trained_model_path=Path(training.model_path),
            params_batch_size=params.BATCH_SIZE,
            params_epochs=params.EPOCHS,
            params_image_size=params.IMAGE_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            updated_model_path=Path(self.config.prepare_base_model.updated_model_path),
            training_data_path=Path(training_data_path)
        )

        return training_config