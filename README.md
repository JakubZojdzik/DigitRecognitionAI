# Digit Recognition AI with Neural Network

## Overview

This repository contains a digit recognition AI implemented using a neural network built from scratch in Python. The neural network has 2 hidden layers, each consisting of 16 neurons. The activation functions used are Rectified Linear Unit (ReLU) for the hidden layers and Softmax for the output layer. Loss function employed is cross entropy.

## TIY (Train It Yourself)

```sh
git clone https://github.com/JakubZojdzik/DigitRecognitionAI.git
cd DigitRecognitionAI
pip install requirements.txt
python main.py
```

## Dataset

The MNIST dataset is used for training and testing the digit recognition model. It can be downloaded from http://yann.lecun.com/exdb/mnist. In order to create some more training data, `Reader` moves images by 1 in different directions and save them as different images.

## Results

The best accuracy I achieved with the model on the MNIST dataset was ~90% after 30min of training.

## TODO

- Saving and loading models