# PyTorch Image Classification using CNN on CIFAR-10

.
├── model/
│   ├── train.py        # Training script for CNN model in PyTorch
│   ├── model.py        # CNN model definition
│   └── resnet.py       # ResNet model for transfer learning
├── requirements.txt    # Required libraries

This project demonstrates image classification using Convolutional Neural Networks (CNNs) in PyTorch. It includes both a custom CNN model and an implementation using transfer learning with a pre-trained ResNet model.

## Table of Contents
0. [Project Overview](#project-overview)
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Architecture](#model-architecture)
5. [Training and Validation](#training-and-validation)
6. [Transfer Learning](#transfer-learning)
7. [Acknowledgments](#acknowledgments)

## 0. Project Overview
This project classifies images from the CIFAR-10 dataset into 10 categories:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

Two model architectures are provided:
1. **Custom CNN**: A custom CNN built from scratch.
2. **ResNet Transfer Learning**: A pre-trained ResNet model fine-tuned for CIFAR-10.

The project allows you to switch between these models and compare performance.

## 1. Requirements

The project requires the following libraries:

- Python 3.x
- torch (PyTorch)
- torchvision
- numpy
- matplotlib

To install the necessary libraries, refer to the [Installation](#installation) section below.

## 2. Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>

```

### 2. Install the Required Packages

You can install the required packages via pip:

```bash
pip install -r requirements.txt
```
Or if you're using conda, you can install via:

```bash
conda install pytorch torchvision torchaudio -c pytorch
```
## 3. Usage

#### 1. Train the Custom CNN Model

To train the CNN model on CIFAR-10, navigate to the `pytorch` folder and run:

```bash
python train.py
```
This will:

* Download the CIFAR-10 dataset (if not already available).
* Train the custom CNN model for a specified number of epochs.
* Output training and validation accuracy after each epoch.
* Save the trained model to `cnn_pytorch.pth`.
#### 2. Train with ResNet for Transfer Learning

To switch to transfer learning with ResNet, modify the `train.py` script to load the ResNet model from `resnet.py`:

```python
from resnet import ResNetTransferLearning
model = ResNetTransferLearning().to(device)
```
Then run the training script again:

```bash
python train.py
```

## 4. Model Architecture

#### 1. Custom CNN Model

The custom CNN model consists of:

- 3 convolutional layers with ReLU activations and max-pooling.
- A fully connected layer with a dropout layer for regularization.
- Final output layer with 10 neurons (for 10 classes).

#### 2. ResNet Transfer Learning

ResNet-18, a pre-trained model from PyTorch's `torchvision.models`, is fine-tuned on CIFAR-10 by replacing the final fully connected layer.

## 5. Training and Validation
* The training script evaluates the model on the test set after each epoch.
* Both loss and accuracy metrics are displayed during training.
* A checkpoint is saved with the model's parameters at the end of training.

## 6. Transfer Learning
Transfer learning allows leveraging pre-trained models for similar tasks, significantly reducing training time and potentially increasing accuracy.

In this project, ResNet-18 is used for transfer learning. The final layer is adjusted for CIFAR-10's 10 output classes.
## 7. Acknowledgements
This project uses the following datasets and libraries:

* CIFAR-10 dataset: A popular dataset used for benchmarking image classification algorithms.
* PyTorch: An open-source machine learning library for Python, used for the model implementation.
