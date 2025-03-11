# Vehicle Collision Prediction Model

This project implements a machine learning model for predicting vehicle collisions using computer vision and feature extraction techniques. The model analyzes vehicle features and predicts collision danger from video frames.

## Project Overview

The system consists of several components:
- Feature extraction from vehicle images
- Collision danger prediction
- Model training and evaluation pipeline
- Dataset preparation and processing tools

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- numpy
- tqdm
- Jupyter Notebook

## Project Structure

```
.
├── data/                      # Data directory for datasets
├── data_preparation.ipynb     # Jupyter notebook for data preprocessing
├── download_gtacrash.py       # Script to download GTA Crash dataset
├── evaluate_predictions.py    # Evaluation metrics and analysis
├── featureExtractor_train.py  # Model training script
└── predict_nexus_features.py  # Inference script for Nexus dataset
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kaggle-collision-prediction.git
cd kaggle-collision-prediction
```

2. Install required dependencies:
```bash
pip install torch torchvision pillow numpy tqdm jupyter
```

3. Download and prepare the datasets:
```bash
python download_gtacrash.py
```

## Usage

### Data Preparation
1. Run the data preparation notebook:
```bash
jupyter notebook data_preparation.ipynb
```

### Training
1. Train the feature extractor model:
```bash
python featureExtractor_train.py
```

### Prediction
1. Run predictions on new data:
```bash
python predict_nexus_features.py
```

### Evaluation
1. Evaluate model performance:
```bash
python evaluate_predictions.py
```

## Model Architecture

The project uses a deep learning model based on a pre-trained backbone with custom layers for feature extraction and collision prediction. The model extracts relevant vehicle features such as position, speed, and trajectory to assess collision risk.

## Data

The project uses two main datasets:
1. GTA Crash Dataset - For training and validation
2. Nexus Dataset - For additional testing and real-world application

## Evaluation Metrics

The model's performance is evaluated using:
- Feature prediction accuracy
- Collision prediction accuracy
- Mean Average Precision (mAP)
- Temporal consistency metrics



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

