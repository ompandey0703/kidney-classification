import os
from zipfile import ZipFile
import urllib.request as request
import tensorflow as tf
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        logger.info("Preparing the base model...")
        # Download weights if needed, else use 'imagenet' or path
        weights = self.config.params_weights
        if weights == "imagenet":
            weights_arg = "imagenet"
        else:
            weights_arg = self.config.base_model_path

        base_model = tf.keras.applications.vgg16.VGG16(
            input_shape=tuple(self.config.params_image_size),
            include_top=self.config.params_include_top,
            weights=weights_arg
        )
        os.makedirs(os.path.dirname(self.config.base_model_path), exist_ok=True)
        base_model.save(self.config.base_model_path)
        
        return base_model

    def _prepare_full_model(self, classes, freeze_all, freeze_till, learning_rate):
        logger.info("Preparing the full model...")
        base_model = self.get_base_model()

        if freeze_all:
            for layer in base_model.layers:
                layer.trainable = False
        elif freeze_till is not None:
            for layer in base_model.layers[:freeze_till]:
                layer.trainable = False

        x = base_model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        output = tf.keras.layers.Dense(classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        return model

    def update_base_model(self):
        logger.info("Updating the base model...")
        model = self._prepare_full_model(
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        os.makedirs(os.path.dirname(self.config.updated_model_path), exist_ok=True)
        model.save(self.config.updated_model_path)
        logger.info(f"Updated model saved at {self.config.updated_model_path}")
