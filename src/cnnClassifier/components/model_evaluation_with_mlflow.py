import tensorflow as tf
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig


class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def load_model(self):
        logger.info(f"Loading model from {self.config.path_of_model}")
        model = tf.keras.models.load_model(self.config.path_of_model)
        logger.info("Model loaded successfully.")
        return model

    def get_data_generator(self):
        from tensorflow.keras.preprocessing import image_dataset_from_directory

        IMG_SIZE = self.config.params_image_size[:2]
        BATCH_SIZE = self.config.params_batch_size
        DATA_DIR = self.config.training_data_path

        val_ds = image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )
        return val_ds

    def evaluate(self):
        model = self.load_model()
        val_ds = self.get_data_generator()
        logger.info("Evaluating model on validation set...")
        results = model.evaluate(val_ds, return_dict=True)
        logger.info(f"Evaluation results: {results}")
        return results
