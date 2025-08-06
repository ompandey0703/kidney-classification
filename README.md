# Kidney Classification AI

A deep learning project for classifying kidney CT scan images into different categories (Normal, Cyst, Tumor, Stone) using VGG16 transfer learning.

## Features

- **AI-powered Classification**: Uses VGG16 pre-trained model for accurate kidney condition classification
- **Web Interface**: Modern, responsive web application for easy image upload and analysis
- **MLflow Integration**: Experiment tracking and model versioning with DagsHub
- **DVC Pipeline**: Data and model versioning with reproducible ML pipelines
- **Docker Support**: Containerized application for easy deployment
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions

## Classes

The model classifies kidney CT scans into 4 categories:
- **Normal**: Healthy kidney tissue
- **Cyst**: Kidney cysts
- **Tumor**: Kidney tumors
- **Stone**: Kidney stones

## Setup and Installation

### Prerequisites
- Python 3.10+
- Git
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ompandey0703/kidney-classification.git
   cd kidney-classification
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv kidney
   source kidney/bin/activate  # On Windows: kidney\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup DVC (optional):**
   ```bash
   dvc init
   dvc remote add -d dagshub https://dagshub.com/ompandey0703/kidney-classification.dvc
   dvc pull  # Pull model and data
   ```

## Usage

### Training Pipeline

Run the complete training pipeline:
```bash
python main.py
```

Or run individual stages:
```bash
# Data ingestion
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py

# Prepare base model
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py

# Model training
python src/cnnClassifier/pipeline/stage_03_training_model.py

# Model evaluation
python src/cnnClassifier/pipeline/stage_04_model_evaluation_with_mlflow.py
```

### Web Application

Start the Flask web application:
```bash
python app.py
```

Visit `http://localhost:8080` in your browser to use the web interface.

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t kidney-classifier .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:8080 kidney-classifier
   ```

## DVC Pipeline

The project uses DVC for data and model versioning:

```bash
# Run the complete pipeline
dvc repro

# Check pipeline status
dvc status

# View pipeline DAG
dvc dag
```

## MLflow Tracking

Experiments are tracked using MLflow and DagsHub:
- View experiments at: https://dagshub.com/ompandey0703/kidney-classification

## Configuration

- **Model Parameters**: Edit `params.yaml`
- **Paths and URLs**: Edit `config/config.yaml`
- **Dependencies**: Edit `requirements.txt`

## Contact

- **Author**: Om Pandey
- **GitHub**: [@ompandey0703](https://github.com/ompandey0703)
- **Project**: [kidney-classification](https://github.com/ompandey0703/kidney-classification)
10. Update the dvc.yaml
11. app.py