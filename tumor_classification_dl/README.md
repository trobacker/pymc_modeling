# Tumor Classification with Deep Learning

A comprehensive deep learning pipeline for binary classification of medical imaging data (benign vs. malignant tumors).

## Project Structure

```
tumor_classification_dl/
├── notebooks/              # Jupyter notebooks with tutorials
├── data/                   # Synthetic and real medical imaging data
├── models/                 # Saved model checkpoints
├── outputs/                # Training logs, plots, and results
├── pyproject.toml          # Project dependencies
└── README.md              # This file
```

## Setup

This project uses `uv` for virtual environment management:

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -e .
```

## Features

- **Synthetic MRI-like data generation** for tumor classification
- **Convolutional Neural Network (CNN)** architecture built with PyTorch
- **Data augmentation pipeline** to improve model robustness
- **Comprehensive evaluation metrics**: accuracy, precision, recall, F1-score, ROC-AUC
- **Interactive dashboards** for classification metric visualization
- **Tutorial notebooks** with detailed explanations

## Quick Start

1. Navigate to `notebooks/tumor_classification_tutorial.ipynb`
2. Run all cells to train a tumor classification model
3. Explore the interactive dashboard for model performance metrics

## Model Architecture

The CNN architecture includes:
- Multiple convolutional layers with ReLU activation
- Max pooling for spatial downsampling
- Batch normalization for stable training
- Dropout for regularization
- Fully connected layers for classification

## Requirements

- Python 3.10+
- PyTorch 2.1.0+
- CUDA-capable GPU (optional, for faster training)
