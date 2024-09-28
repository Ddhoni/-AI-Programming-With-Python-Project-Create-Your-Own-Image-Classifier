# AI Programming with Python: Image Classifier Project

## Overview
This project aims to integrate AI algorithms into everyday applications by training an image classifier to identify different species of flowers. The classifier can be utilized within a smartphone app to help users identify flowers in real-time. The model is trained on a dataset containing 102 flower categories and is designed to be exported for use within other applications.

## Prerequisites
- **Programming Language**: Python 3.6.5
- **Required Libraries**: Numpy, Pandas, Matplotlib, Pytorch, PIL, json
- **Installation**: Visit the official Pytorch website for detailed installation instructions.

## Usage

### Training the Network
- **Basic Command**: `python train.py data_directory`
  - This command initiates training and outputs the epoch number, training loss, validation loss, and validation accuracy.
- **Options**:
  - Save checkpoints: `python train.py data_dir --save_dir save_directory`
  - Choose model architecture: `python train.py data_dir --arch "vgg19"`
  - Set hyperparameters: `python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20`
  - Enable GPU training: `python train.py data_dir --gpu`

### Predicting with the Network
- **Basic Command**: `python predict.py /path/to/image checkpoint`
  - Outputs the predicted flower name and its probability.
- **Options**:
  - Top K predictions: `python predict.py input checkpoint --top_k 3`
  - Mapping categories to names: `python predict.py input checkpoint --category_names cat_to_name.json`
  - Enable GPU inference: `python predict.py input checkpoint --gpu`

## GPU Considerations
Training deep convolutional neural networks require significant computational power. Alternatives include:
- **CUDA**: Suitable for NVIDIA GPU users, although training can still be lengthy.
- **Cloud Services**: AWS, Google Cloud, and other services offer powerful computing resources for training.
- **Google Colab**: Provides free access to a Tesla K80 GPU for 12 hours, suitable for datasets that can fit within Google Drive's storage limits.

## Data and JSON
- A .json file is required for the network to correctly identify flower names from numerical folder labels.
- The data structure should consist of three folders: train, test, and validate, with subfolders representing specific categories as defined in the .json file.

## Hyperparameters
Choosing effective hyperparameters can significantly impact the performance of the neural network:
- Increasing epochs can improve accuracy but may hinder generalization.
- A high learning rate can cause the model to overshoot the minimal error.
- A low learning rate leads to higher accuracies but requires more time.
- Densenet121 is optimal for image processing but requires more training time compared to other architectures like alexnet or vgg19.

## Pre-Trained Network
The `checkpoint.pth` file contains a model trained on 102 flower species with specific hyperparameters. Use the following command for predictions with the pretrained model: `python predict.py /path/to/image checkpoint.pth`. Note: This file is not included in the repository.
