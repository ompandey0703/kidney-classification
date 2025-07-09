from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation_with_mlflow import ModelEvaluationPipeline


STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(f"An error occurred in the {STAGE_NAME} stage: {e}")
    raise e

STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    prepare_base_model_pipeline = PrepareBaseModelPipeline()
    prepare_base_model_pipeline.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"An error occurred in the {STAGE_NAME} stage: {e}")
    raise e

STAGE_NAME = "Model Training"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"An error occurred in the {STAGE_NAME} stage: {e}")
    raise e

STAGE_NAME = "Model Evaluation"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"An error occurred in the {STAGE_NAME} stage: {e}")
    raise e
