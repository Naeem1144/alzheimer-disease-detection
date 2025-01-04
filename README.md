# Alzheimer's Disease Classification with Deep Learning

![Project Banner Image](https://github.com/Naeem1144/Alzheimer_Prediction_CNN/blob/main/Images/Banner.png)

## Overview

This project focuses on the classification of Alzheimer's disease using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The goal is to accurately classify brain MRI images into four categories: NonDemented, VeryMildDemented, MildDemented, and ModerateDemented. Early and accurate diagnosis of Alzheimer's disease is crucial for effective treatment and patient care. This project demonstrates the potential of deep learning in assisting medical professionals in this task.

The model developed in this project achieved a **validation accuracy of 99.21%** and a **test accuracy of 99%**, showcasing the effectiveness of CNNs in medical image classification.

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
*   **Epochs:** 100 (with early stopping).
*   **Batch Size:** 32.
*   **Device:** GPU (if available, otherwise CPU).
*   **Dataset Split:** 70% training, 15% validation, 15% testing.

## Results

The model was trained for 61 epochs, with early stopping triggered when the validation accuracy did not improve for 7 consecutive epochs. The best model was saved at epoch 54, achieving a validation accuracy of 99.21%.

**Training Summary:**

| Metric             | Value    |
| :----------------- | :------- |
| Best Epoch        | 54       |
| Validation Accuracy | 99.21% |
| Test Accuracy      | 99.00%    |
| Total Training Time | 20791.99 seconds |


**Confusion Matrix (Validation):**

![Confusion Matrix Validation](https://github.com/Naeem1144/Alzheimer_Prediction_CNN/blob/main/Images/Confusion-Matrix-validation.png)

**Confusion Matrix (Test):**

![Confusion Matrix Test](https://github.com/Naeem1144/Alzheimer_Prediction_CNN/blob/main/Images/Confusion-Matrix-test.png)

**Training Loss Curve:**

![Training Loss Curve](https://github.com/Naeem1144/Alzheimer_Prediction_CNN/blob/main/Images/Training-Curve.png)

**Validation Accuracy Curve:**

![Validation Accuracy Curve](https://github.com/Naeem1144/Alzheimer_Prediction_CNN/blob/main/Images/Validation-Accuracy.png)

*Note: Replace the image links with the actual links to your saved images or upload them to your repository.*

## Dependencies

*   Python 3.11.9
*   PyTorch
*   torchvision
*   scikit-learn
*   pandas
*   matplotlib
*   seaborn
*   Pillow
*   NumPy

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```

2. Navigate to the project directory:

    ```bash
    cd your-repo-name
    ```

3. Install the required dependencies (preferably in a virtual environment):

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Download the dataset** from the provided Kaggle link and place it in the appropriate directory (as per the notebook's paths).
2. **Run the Jupyter Notebook** `alzheimer-classification.ipynb` to train the model and evaluate its performance.

## Notebook Structure

*   **Data Loading and Preprocessing:** Includes data augmentation and dataset splitting.
*   **Model Definition:** Defines the `AlzheimerCNN` architecture.
*   **Training Loop:** Implements the training and validation process with early stopping.
*   **Model Evaluation:** Evaluates the trained model on the test set, including the generation of a confusion matrix and classification report.
*   **Visualization:** Includes plots of training loss and validation accuracy.
*   **Probability Distribution:** The notebook contains a function `get_output` to visualize the probability distribution of the model's predictions for a given input image.

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

*   The dataset used in this project is from Kaggle's "Well-Documented Alzheimer's Dataset."
*   Inspired by various deep learning tutorials and the PyTorch documentation.


