# Alzheimer's Disease Classification with Deep Learning

![Project Banner Image](https://github.com/Naeem1144/Alzheimer_Prediction_CNN/blob/main/Images/Banner.png)

## Overview

This project focuses on the classification of Alzheimer's disease using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The goal is to accurately classify brain MRI images into four categories: NonDemented, VeryMildDemented, MildDemented, and ModerateDemented. Early and accurate diagnosis of Alzheimer's disease is crucial for effective treatment and patient care. This project demonstrates the potential of deep learning in assisting medical professionals in this task.

The model developed in this project achieved a **validation accuracy of 99.21%** and a **test accuracy of 99%**, showcasing the effectiveness of CNNs in medical image classification.

here is gui application of this deep learning model (upload your own image OR select get it form the [Example Dataset](https://github.com/Naeem1144/Alzheimer_Prediction_CNN/tree/main/Examples%20Images) : <https://alzheimerpredictioncnn-naeem.streamlit.app/>

## Dataset

The project utilizes the **Well-Documented Alzheimer's Dataset** from Kaggle, which can be found [here](https://www.kaggle.com/datasets/yiweilu2033/well-documented-alzheimers-dataset).

**Dataset Details:**

*   **Source:** Kaggle

*   **Title:** Well-Documented Alzheimer's Dataset

*   **URL:** [https://www.kaggle.com/datasets/yiweilu2033/well-documented-alzheimers-dataset](https://www.kaggle.com/datasets/yiweilu2033/well-documented-alzheimers-dataset)

*   **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International)

*   **Classes:**
    *   NonDemented
    *   VeryMildDemented
    *   MildDemented
    *   ModerateDemented

*   **Image Format:** The dataset consists of MRI images.

## Data Preprocessing and Augmentation

The following data preprocessing and augmentation techniques were applied to enhance the model's performance and generalization:

1. **Resizing:** Images were resized to 256x256 pixels.
2. **Random Horizontal Flip:** Images were randomly flipped horizontally with a probability of 0.5.
3. **Random Rotation:** Images were randomly rotated by up to 15 degrees.
4. **Color Jitter:** Random adjustments were made to brightness, contrast, saturation, and hue.
5. **Random Affine:** Random affine transformations (translation) were applied.
6. **Normalization:** Images were normalized using the ImageNet mean and standard deviation.

## Model Architecture

The deep learning model used in this project is a custom Convolutional Neural Network (CNN) defined by the `AlzheimerCNN` class.

## Training

*   **Optimizer:** Adam optimizer with a learning rate of 0.0001 and weight decay of 1e-5.
*   **Loss Function:** Cross-Entropy Loss.
*   **Scheduler:** ReduceLROnPlateau scheduler with mode='max', factor=0.1, patience=2.
