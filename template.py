from pathlib import Path
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


project_name = "cnnClassifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/entity/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "dvc.yaml",
    "research/trial.ipynb",
    "templates/index.html"]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = os.path.dirname(str(filepath))
    
    if filedir and not os.path.exists(filedir):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write("# Placeholder for " + str(filepath))
        logging.info(f"Created file: {filepath}")
    else:
        logging.warning(f"File already exists: {filepath}")

logging.info("Project structure created successfully.")