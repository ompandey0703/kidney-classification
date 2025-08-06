import os
import zipfile
import gdown
from cnnClassifier.utils.common import get_size_of_file
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the dataset from the specified URL and saves it to the local directory.
        """
        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            if not os.path.exists(zip_download_dir):
                logger.info(
                    f"Downloading file from {self.config.source_url} to {self.config.local_data_file}")

                prefix = "https://drive.google.com/uc?/export=download&id="
                file_id = dataset_url.split("/")[-2]
                gdown.download(
                    prefix+file_id, str(zip_download_dir), quiet=False)
                logger.info(
                    f"Downloaded file size: {get_size_of_file(self.config.local_data_file)} bytes")
            else:
                logger.info(f"Zip file already exists")
        except Exception as e:
            logger.error(f"Error occurred while downloading file: {e}")
            raise e

    def extract_zip_file(self):
        logger.info(
            f"Extracting zip file {self.config.local_data_file} to {self.config.unzip_dir}")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info("Extraction completed")
