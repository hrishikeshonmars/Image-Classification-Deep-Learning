# Image-Classification-Deep-Learning

Image Classification Model
This repository contains code for developing a machine learning model for image classification. The goal is to accurately classify images into different categories based on their content. The model is trained on a provided dataset containing approximately 25,000 images distributed across six categories: buildings, forest, glacier, mountain, sea, and street.

Installation
Clone the repository to your local machine:
git clone https://github.com/hrishikeshonmars/image-classification-Deep-Learning.git

Navigate to the project directory:
cd image-classification-Deep-Learning

Install the required dependencies using pip:
pip install -r requirements.txt

Ensure you have Python installed on your system. The code is written in Python 3.
Dataset
The dataset is divided into three sets: train, validation, and test.
Train set: Approximately 14,000 images.
Validation set: Approximately 7,000 images.
Test set: Approximately 3,000 images.
Each image is labeled with the corresponding category (0 for buildings, 1 for forest, 2 for glacier, 3 for mountain, 4 for sea, and 5 for street).

Model Architecture
The model architecture is based on a convolutional neural network (CNN) for image classification. It consists of the following layers:
Convolutional layers with ReLU activation.
Max pooling layers.
Flatten layer.
Dense layers with ReLU activation for classification.
Softmax activation for the final output layer.

Hyperparameters
Convolutional layers: 32 filters, kernel size (3, 3).
Dense layers: 128 units with ReLU activation.
Output layer: 6 units with softmax activation.
Optimizer: Adam optimizer.
Loss function: Sparse categorical cross-entropy.
Metrics: Accuracy.

Data Preprocessing
Resized images to (150, 150) pixels.
Normalized pixel values to range [0, 1].
Utilized data augmentation techniques to increase the diversity of the training set.

Training and Evaluation
Trained the model on the training set.
Evaluated the model's performance on the validation set.
Optimized the model and hyperparameters to achieve the best performance.
Finally, evaluated the model on the test set to assess its generalization ability.

Usage
To run the code, follow these steps:
Ensure you have Python and necessary libraries installed (TensorFlow, NumPy, pandas, scikit-learn).
Clone this repository to your local machine.
Install dependencies using pip install -r requirements.txt.

Run the main Python script:
python image_classification.py

Contributor
Hrishikesh KA
