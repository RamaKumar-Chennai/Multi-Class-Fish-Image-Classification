# Multi-Class-Fish-Image-Classification

Project Overview
The objective of this project is to classify fish species using deep learning models. It compares the performance of a basic CNN and several pretrained models to identify which architecture performs best for fish image classification.

Dataset: Images labeled into 11 fish categories.
Techniques: Custom CNN, data augmentation, pretrained model fine-tuning (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).
Evaluation: Metrics include accuracy, precision, recall, F1-score, and confusion matrix.
Installation
pip install torch torchvision matplotlib numpy pandas seaborn scikit-learn tqdm
pip install streamlit pyngrok streamlit-folium nbconvert
npm install localtunnel
Imports
# File Handling and Image Processing
import os
import zipfile
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random

# Torch and Vision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Display and Progress
from IPython.display import display
from tqdm import tqdm

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Warnings
import warnings
warnings.filterwarnings('ignore')



Dataset
The dataset contains labeled fish images categorized into:

train/: Training data
val/: Validation data
test/: Testing data
Directory Structure
Dataset/
├── train/
│   ├── class1/
│   ├── class2/
├── val/
│   ├── class1/
│   ├── class2/
├── test/
│   ├── class1/
│   ├── class2/
Preprocessing
Resize images to 224x224
Random horizontal flips
Rotation between -15° to 15°
Random affine transforms
Normalize using ImageNet mean and std
Model
CNN Model
2 convolutional layers + ReLU + MaxPooling
Dropout layers
Fully connected layers for classification
Pretrained Models
VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
Custom final layers adapted to 11 classes
vgg16.classifier[6] = nn.Linear(num_features, 11)
Loss and Optimizer
Loss: CrossEntropyLoss
Optimizer: Adam (lr=0.001)
Training
Trained for 20 epochs
Batch size: 32
Used train_loader and val_loader for training and validation
DataLoader with shuffle=True for training
Model checkpointing and saving best .pth model
Each pretrained model was trained using:

train_loader for training
val_loader for validation
Batch size = 32
Evaluation done on eval_loader or test_loader
Metrics like accuracy, precision, recall, and F1 were computed per model
Evaluation
Metrics used:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Results
Model Performance Comparison
Model	Accuracy	Precision	Recall	F1 Score
VGG16	85.5%	84.0%	85.0%	84.5%
ResNet50	87.2%	86.5%	87.0%	86.7%
MobileNet	89.0%	88.0%	89.2%	88.6%
InceptionV3	90.3%	89.5%	90.0%	89.7%
EfficientNetB0	91.1%	90.5%	91.2%	90.9%
Usage
Run the Streamlit app from Colab or local:

!streamlit run /content/app.py &>/content/logs.txt & npx localtunnel --port 8501
Class Names
The 11 fish species classes used in the model are:

animal fish
animal fish bass
fish sea_food black_sea_sprat
fish sea_food gilt_head_bream
fish sea_food hourse_mackerel
fish sea_food red_mullet
fish sea_food red_sea_bream
fish sea_food sea_bass
fish sea_food shrimp
fish sea_food striped_red_mullet
fish sea_food trout
