import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        model = tf.keras.models.load_model(self.config.updated_model_path)
        return model

    def get_data_generators(self):
        # ✅ Folder-based loading using image_dataset_from_directory
        from tensorflow.keras.preprocessing import image_dataset_from_directory

        IMG_SIZE = self.config.params_image_size[:2]
        BATCH_SIZE = self.config.params_batch_size
        # This should be the dataset folder path
        DATA_DIR = Path(self.config.training_data_path)

        train_ds = image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )

        val_ds = image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )

        # Optional: prefetch for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds

    def save_model(self):
        self.model.save(self.config.train_model.model_path)
        logger.info("Training completed and model saved")

    def train(self):
        logger.info("Training started")

        # ❌ No need to load CSV or dataframe
        # ✅ Instead, load folder-based datasets
        train_ds, val_ds = self.get_data_generators()
        sample_batch = next(iter(train_ds))
        print("Train batch shape:", sample_batch[0].shape)

        # Build model
        self.model = self.get_base_model()

        # self.model.compile(optimizer='adam',
        #                    loss='categorical_crossentropy',
        #                    metrics=['accuracy'])

        # Train the model
        self.model.fit(
            train_ds, validation_data=val_ds, epochs=self.config.params_epochs
        )

        # Save the trained model
        self.save_model()
