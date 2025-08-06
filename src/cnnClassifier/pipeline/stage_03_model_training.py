

from cnnClassifier import logger
from cnnClassifier.components.model_training import Training
from cnnClassifier.config.configuration import ConfigurationManager
STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    """
    Pipeline for training the model.
    This stage is responsible for training the model using the prepared base model and the training dataset.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            training_config = config_manager.get_training_config()
            training = Training(training_config)
            training.train()
        except Exception as e:
            logger.exception(f"Exception occurred during training: {e}")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        config_manager = ConfigurationManager()
        model_training_pipeline = ModelTrainingPipeline(config=config_manager)
        model_training_pipeline.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in the Model Training stage: {e}")
        raise e
