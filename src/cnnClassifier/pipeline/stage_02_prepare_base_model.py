from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelPipeline:
    """
    Pipeline for preparing the base model.
    This stage is responsible for setting up the base model configuration and ensuring
    that all necessary components are ready for training.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            prepare_base_model_config = config_manager.get_prepare_base_model_config()

            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        config_manager = ConfigurationManager()
        prepare_base_model_pipeline = PrepareBaseModelPipeline(config=config_manager)
        prepare_base_model_pipeline.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in the Prepare Base Model stage: {e}")
        raise e
