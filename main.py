from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline

# STAGE_NAME = "Data Ingestion"

# try:
#     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
#     data_ingestion_pipeline = DataIngestionPipeline()
#     data_ingestion_pipeline.main()
#     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
# except Exception as e:
#     logger.exception(f"An error occurred in the {STAGE_NAME} stage: {e}")
#     raise e

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
    from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"An error occurred in the {STAGE_NAME} stage: {e}")
    raise e
