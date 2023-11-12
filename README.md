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
