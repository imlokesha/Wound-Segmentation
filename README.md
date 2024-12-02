# Wound Segmentation Project

## Overview
This project implements a deep learning-based solution for automated wound segmentation in medical images. The system uses Convolutional Neural Networks (CNN) to identify and segment wound regions from input images, helping healthcare professionals in wound assessment and monitoring.

## Features
- Automated wound region detection and segmentation
- Support for various image formats
- Data preprocessing pipeline with automatic background removal
- Optional data augmentation for improved model training
- Flexible processing modes for both training and inference


```bash
pip install tensorflow numpy tqdm opencv-python
```

## Project Structure

```
WoundSegmentation/
│
├── data/
│   ├── images/          # Input images
│   └── labels/          # Segmentation masks
│
├── src/
│   ├── cnn.ipynb        # Main notebook with model implementation
│   └── utils/           # Utility functions
│
└── README.md
```

## PUsage
### Data Preparation
Place your wound images in the data/images directory and corresponding segmentation masks in data/labels. Images and their masks should have matching filenames.

### Training Data Processing

```python
from cnn import get_data

# Load and preprocess training data with augmentation
train_images, train_masks = get_data(
    path='path/to/data',
    labels_present=True,
    augment=True
)
```

## Inference Mode
```python
# Process new images without labels
test_images = get_data(
    path='path/to/test/data',
    labels_present=False,
    augment=False
)
```

## Data Processing Pipeline
1. Image Loading: Loads images from specified directory

2. Background Removal: Automatically crops images to remove unnecessary background

3. Preprocessing:
    - Resizing using Lanczos interpolation

    - Normalization to 0,1 range

    - Optional data augmentation

    - Mask binarization (for training data)

## Configuration Options
- labels_present: Toggle between training and inference modes

- augment: Enable/disable data augmentation

- change_mask: Option to scale mask values

- Image dimensions and other parameters can be modified in the configuration section

