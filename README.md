# Fish Species Classification with Convolutional Neural Networks (CNN)

## Overview

This project involves the classification of nine different species of fishes using Convolutional Neural Networks (CNN). The aim is to develop a robust deep learning model that can accurately identify the species of a fish given an image.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The classification of fish species is a crucial task in marine biology and fisheries management. Automated classification using CNNs can significantly aid in monitoring and managing fish populations. This project leverages deep learning techniques to classify images of nine fish species.

## Dataset

The dataset used in this project consists of images belonging to the following nine species of fish:

1. Black Sea Sprat
2. Gilt Head Bream
3. Horse Mackerel
4. Red Mullet
5. Red Sea Bream
6. Sea Bass
7. Shrimp
8. Striped Red Mullet
9. Trout

Each species has a significant number of images to train and test the model effectively.

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/rdzhulai/fish-classification.git
    cd fish-classification
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use this project, follow these steps:

1. Prepare the dataset and place it in the `data/` directory.
2. Open and run the cells in the `train_test_model.ipynb` notebook sequentially.
3. Train the model by executing the relevant cells in the notebook.
4. Evaluate the model's performance by running the evaluation cells.
5. Predict species for new images by executing the prediction cells with your image paths.

## Model Architecture

The CNN model used in this project is a custom architecture consisting of several Residual Blocks followed by fully connected layers. The model architecture is summarized below:

```plaintext
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CustomModel                              [5, 9]                    --
├─Sequential: 1-1                        [5, 512, 44, 33]          --
│    └─ResBlock: 2-1                     [5, 64, 708, 532]         --
│    │    └─Conv2d: 3-1                  [5, 64, 708, 532]         1,792
│    │    └─BatchNorm2d: 3-2             [5, 64, 708, 532]         128
│    │    └─ReLU: 3-3                    [5, 64, 708, 532]         --
│    │    └─Conv2d: 3-4                  [5, 64, 708, 532]         36,928
│    │    └─BatchNorm2d: 3-5             [5, 64, 708, 532]         128
│    │    └─Sequential: 3-6              [5, 64, 708, 532]         384
│    │    └─ReLU: 3-7                    [5, 64, 708, 532]         --
│    └─MaxPool2d: 2-2                    [5, 64, 354, 266]         --
│    └─ResBlock: 2-3                     [5, 128, 354, 266]        --
│    │    └─Conv2d: 3-8                  [5, 128, 354, 266]        73,856
│    │    └─BatchNorm2d: 3-9             [5, 128, 354, 266]        256
│    │    └─ReLU: 3-10                   [5, 128, 354, 266]        --
│    │    └─Conv2d: 3-11                 [5, 128, 354, 266]        147,584
│    │    └─BatchNorm2d: 3-12            [5, 128, 354, 266]        256
│    │    └─Sequential: 3-13             [5, 128, 354, 266]        8,576
│    │    └─ReLU: 3-14                   [5, 128, 354, 266]        --
│    └─MaxPool2d: 2-4                    [5, 128, 177, 133]        --
│    └─ResBlock: 2-5                     [5, 256, 177, 133]        --
│    │    └─Conv2d: 3-15                 [5, 256, 177, 133]        295,168
│    │    └─BatchNorm2d: 3-16            [5, 256, 177, 133]        512
│    │    └─ReLU: 3-17                   [5, 256, 177, 133]        --
│    │    └─Conv2d: 3-18                 [5, 256, 177, 133]        590,080
│    │    └─BatchNorm2d: 3-19            [5, 256, 177, 133]        512
│    │    └─Sequential: 3-20             [5, 256, 177, 133]        33,536
│    │    └─ReLU: 3-21                   [5, 256, 177, 133]        --
│    └─MaxPool2d: 2-6                    [5, 256, 88, 66]          --
│    └─ResBlock: 2-7                     [5, 512, 88, 66]          --
│    │    └─Conv2d: 3-22                 [5, 512, 88, 66]          1,180,160
│    │    └─BatchNorm2d: 3-23            [5, 512, 88, 66]          1,024
│    │    └─ReLU: 3-24                   [5, 512, 88, 66]          --
│    │    └─Conv2d: 3-25                 [5, 512, 88, 66]          2,359,808
│    │    └─BatchNorm2d: 3-26            [5, 512, 88, 66]          1,024
│    │    └─Sequential: 3-27             [5, 512, 88, 66]          132,608
│    │    └─ReLU: 3-28                   [5, 512, 88, 66]          --
│    └─MaxPool2d: 2-8                    [5, 512, 44, 33]          --
├─Sequential: 1-2                        [5, 9]                    --
│    └─Linear: 2-9                       [5, 512]                  380,633,600
│    └─ReLU: 2-10                        [5, 512]                  --
│    └─BatchNorm1d: 2-11                 [5, 512]                  1,024
│    └─Dropout: 2-12                     [5, 512]                  --
│    └─Linear: 2-13                      [5, 9]                    4,617
==========================================================================================
Total params: 385,503,561
Trainable params: 385,503,561
Non-trainable params: 0
Total mult-adds (G): 398.19
==========================================================================================
Input size (MB): 22.60
Forward/backward pass size (MB): 10838.24
Params size (MB): 1542.01
Estimated Total Size (MB): 12402.86
==========================================================================================
```

## Training

The training process involves:

1. Loading and preprocessing the dataset.
2. Defining the CNN model architecture.
3. Compiling the model with appropriate loss function and optimizer.
4. Training the model on the training data.
5. Validating the model on the validation data.

Training parameters such as learning rate, batch size, and number of epochs can be adjusted in the `config.py` file.

## Evaluation

The model is evaluated using the test dataset to measure its accuracy and other performance metrics such as precision, recall, and F1-score. The evaluation script generates a detailed report and confusion matrix.

## Results

The trained model achieves the following performance metrics on the test dataset:

- Accuracy: 79.07%
- Precision: 82.23%
- Recall: 80.00%
- F1-Score: 78.86%

Detailed results and plots can be found in the `results/` directory.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please make sure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to customize this README file further to match the specifics of your project.
