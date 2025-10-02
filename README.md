# Image Classification with CIFAR-10

## Description
This project builds, trains, and evaluates a simple Convolutional Neural Network (CNN) for image classification. It uses the popular CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes (e.g., airplane, dog, cat). The entire process, from data loading to model training and evaluation, is handled within a single script using the PyTorch library.

The `torchvision` library automatically handles downloading the CIFAR-10 dataset, so no manual data setup is required.

## Features
- Simple CNN architecture suitable for beginners.
- Uses PyTorch for model building and training.
- Automatic dataset download and preprocessing.
- Training and evaluation loops.
- Displays a sample of test images with their predicted labels.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Image_Classification_CIFAR10/
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the script:**
    ```bash
    python src/main.py
    ```
    *Note: The first run will download the CIFAR-10 dataset (approx. 163 MB), which may take a few minutes.*

## Example Output
```
Files already downloaded and verified
Files already downloaded and verified
Starting training for 5 epochs...
Epoch 1/5, Loss: 1.7542
Epoch 2/5, Loss: 1.4128
Epoch 3/5, Loss: 1.2855
Epoch 4/5, Loss: 1.1963
Epoch 5/5, Loss: 1.1235
Finished Training.
---
Calculating accuracy on 10000 test images...
Accuracy of the network on the 10000 test images: 61 %
---
Displaying sample predictions...
GroundTruth:  cat | ship | ship | airplane
Predicted:    cat | ship | ship | airplane
(A matplotlib window will open showing the images)
```
