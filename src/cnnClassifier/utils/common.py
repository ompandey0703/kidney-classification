# common.py: To write common functions that can be used in the project
import os
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns i

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises
        ValueError: If the file does not exist or is not a YAML file.
        e: empty file

    Returns:
    ConfigBox: A ConfigBox object containing the YAML data.
    """
    if not path_to_yaml.exists():
        raise ValueError(f"File {path_to_yaml} does not exist.")

    if path_to_yaml.suffix != ".yaml":
        raise ValueError(f"File {path_to_yaml} is not a YAML file.")

    with open(path_to_yaml, "r") as yaml_file:
        try:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise ValueError("YAML file is empty.")
            return ConfigBox(content)
        except yaml.YAMLError as e:
            logger.error(f"Error reading YAML file: {e}")
            raise e


def create_directories(path_to_directories: list) -> None:
    """
    Creates directories if they do not exist.

    Args:
        path_to_directories (list): List of paths to directories (str or Path).
    """
    for path in path_to_directories:
        os.makedirs(str(path), exist_ok=True)
        logger.info(f"Created directory: {path}")


@ensure_annotations
def save_json(path_to_json: Path, data: dict) -> None:
    """Saves a dictionary to a JSON file.

    Args:
        path_to_json (Path): Path to the JSON file.
        data (dict): Data to be saved.
    """
    with open(path_to_json, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"Saved data to {path_to_json}")


@ensure_annotations
def load_json(path_to_json: Path) -> ConfigBox:
    """Loads a dictionary from a JSON file.

    Args:
        path_to_json (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data as class attributes instead of dictionary.
    """
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    logger.info(f"Loaded data from {path_to_json}")
    return ConfigBox(data)


@ensure_annotations
def save_bin(data: Any, path_to_bin: Path) -> None:
    """Saves data to a binary file using joblib.

    Args:
        data (Any): Data to be saved.
        path_to_bin (Path): Path to the binary file.
    """
    joblib.dump(data, path_to_bin)
    logger.info(f"Saved data to {path_to_bin}")


@ensure_annotations
def load_bin(path_to_bin: Path) -> Any:
    """Loads data from a binary file using joblib.

    Args:
        path_to_bin (Path): Path to the binary file.

    Returns:
        Any: Data loaded from the binary file.
    """
    data = joblib.load(path_to_bin)
    logger.info(f"Loaded data from {path_to_bin}")
    return data


@ensure_annotations
def get_size_of_file(file_path: Path) -> int:
    """Returns the size of a file in bytes.

    Args:
        file_path (Path): Path to the file.

    Returns:
        int: Size of the file in bytes.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    size = file_path.stat().st_size
    logger.info(f"Size of {file_path}: {size} bytes")
    return size


def decodeImage(image_string: str) -> bytes:
    """Decodes a base64 encoded image string to bytes.

    Args:
        image_string (str): Base64 encoded image string.

    Returns:
        bytes: Decoded image bytes.
    """
    try:
        decoded_image = base64.b64decode(image_string)
        logger.info("Image decoded successfully.")
        return decoded_image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise e


def encodeImage(image_bytes: bytes) -> str:
    """Encodes image bytes to a base64 string.

    Args:
        image_bytes (bytes): Image bytes to be encoded.

    Returns:
        str: Base64 encoded image string.
    """
    try:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        logger.info("Image encoded successfully.")
        return encoded_image
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise e
