from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_with_mlflow import (
    ModelEvaluation
)
from cnnClassifier import logger
import mlflow
import dagshub

STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    """
    Pipeline for evaluating the trained model and logging metrics to MLflow.
    """

    def __init__(self, config=None):
        self.config = config or ConfigurationManager()

    def main(self):
        try:
            # Initialize dagshub/MLflow connection
            dagshub.init(
                repo_owner="ompandey0703",
                repo_name="kidney-classification",
                mlflow=True,
            )

            eval_config = self.config.get_evaluation_config()
            evaluator = ModelEvaluation(config=eval_config)
            results = evaluator.evaluate()
            logger.info(f"Model evaluation results: {results}")

            # Log metrics to MLflow
            with mlflow.start_run(run_name="model-evaluation"):
                # Log all params from all_params dict
                for k, v in eval_config.all_params.items():
                    mlflow.log_param(k, v)

                for k, v in results.items():
                    mlflow.log_metric(k, v)

        except Exception as e:
            logger.exception(
                f"Exception occurred during model evaluation: {e}")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        config_manager = ConfigurationManager()
        pipeline = ModelEvaluationPipeline(config=config_manager)
        pipeline.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in the {STAGE_NAME} stage: {e}")
        raise e
