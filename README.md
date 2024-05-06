# Deep-Fake-Detection
This Python script uses PyTorch, a machine learning library, to build and train a model to detect deepfake images. The model leverages pretrained neural networks and custom datasets to achieve high accuracy.

Requirements
    Python 3.6+
    PyTorch
    Torchvision
    tqdm

You can install the necessary libraries using pip:

```bash
  pip3 install torch torchvision tqdm
```

Setup
  Prepare Your Dataset: Organize your dataset into three directories: Training, Validation, and Testing. Each directory should contain images in   folders labeled by class.

  Configure Paths: Set the AllData variable to the root path where your dataset directories are located.

Usage

Run the script from your terminal:

```bash
python3 DeepFakeDetection.py
```
Features
    Data Preprocessing: Includes image resizing, normalization, and augmentation (random flips and rotations) to enhance model robustness.
    Device Compatibility: Automatically uses the CPU but can be configured to run on a GPU for faster training.
    Progress Monitoring: Implements tqdm for tracking training progress.

Structure of the Script
    Imports and Setup: Import necessary libraries and define paths.
    Data Transformations: Setup image transformations for training and validation.
    Model Training and Validation: Code for training the model on the dataset, including loading data, setting up the model, training, and   validating the performance.
