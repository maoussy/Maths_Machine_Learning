# Maths_Machine_Learning
#MNIST.ipynb

Data Loading and Preprocessing
Dataset: MNIST, a popular dataset for handwritten digit recognition.
Train/Test Split: The dataset is divided into training and testing sets.
Batch Size: The training batch size is set to 64, and the testing batch size is 1000.
Data Loader: Utilizes PyTorch's DataLoader for efficient data handling.
Preprocessing: The data is transformed into tensors and normalized using mean 0.1307 and standard deviation 0.3081.
Network Architecture
Type: Convolutional Neural Network (CNN).
Layers:
Two convolutional layers (conv1 and conv2).
Dropout layer (conv2_drop) for regularization.
Two fully connected layers (fc1 and fc2).
Training Setup
Epochs: The model is trained for 5 epochs.
Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and momentum 0.5.
Loss Function: Negative Log-Likelihood (NLL) Loss.
Random Seed: Set to 1 for reproducibility.
CUDA Support: Disabled (torch.backends.cudnn.enabled set to False).
Training Process
The train function handles the training process for each epoch.
In each batch, the model's gradients are zeroed, the forward pass is computed, the loss is calculated, and backpropagation is performed.
Training loss is printed every 10 batches.
Testing/Evaluation
The test function evaluates the model on the test dataset.
It computes the total loss and the number of correct predictions to calculate the accuracy.
The model's performance (loss and accuracy) on the test set is printed after each epoch.
Observations
The script includes code to visualize six test images along with their true labels.
The network's architecture is relatively simple, making it suitable for the MNIST dataset, which doesn't require complex models.

#MNIST Cluster

This report presents an analysis of a Python script that uses the k-means clustering algorithm for digit recognition on the MNIST dataset. The script involves loading the dataset, preprocessing it, applying the MiniBatchKMeans clustering algorithm from scikit-learn, and then evaluating the performance of the model. The key aspects of the script are outlined below:

Data Loading and Preprocessing
Dataset: MNIST, which contains 60,000 training and 10,000 testing grayscale images of handwritten digits (0-9).
Data Shapes:
Training data: 60,000 images of 28x28 pixels.
Training labels: 60,000 labels.
Testing data: 10,000 images of 28x28 pixels.
Testing labels: 10,000 labels.
Visualization: The first few images in the training set are visualized along with their corresponding labels.
Data Reshaping and Normalization
The images are reshaped into a 2D array (784 features per image).
Pixel values are normalized to the range [0, 1] by dividing by 255.
K-Means Clustering
MiniBatchKMeans from scikit-learn is used for clustering. Different numbers of clusters (10, 16, 36, 64, 144, 256) are tested.
The script includes two custom functions: infer_cluster_labels and infer_data_labels to associate cluster labels with the true labels of the digits.
Model Evaluation and Results
For each number of clusters, the script calculates the model's inertia, homogeneity score, and accuracy.
The accuracy improves as the number of clusters increases, reaching approximately 89.1% with 256 clusters.
Testing on Unseen Data
The trained k-means model with 256 clusters is tested on the MNIST test dataset.
The test accuracy is also calculated and printed.
Visualization of Cluster Centroids
The centroids of the clusters are reshaped back into 28x28 pixel images and visualized.
Each centroid image represents what the k-means algorithm has learned to be the representative image of that cluster.
Observations
Cluster Homogeneity: As the number of clusters increases, the homogeneity of the clusters improves, indicating that the model is better able to distinguish between different digits.
Accuracy: The accuracy improves significantly with the number of clusters, showing that more clusters allow for a finer distinction between different types of digits.
Model Limitations: Despite the high accuracy, k-means has inherent limitations as a clustering algorithm for image classification, particularly in its assumption of cluster shapes and the use of Euclidean distance.
