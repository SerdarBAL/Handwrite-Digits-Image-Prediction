import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np


# Function to plot model's learned parameters (weights)
def PlotParameters(model):
    W = model.state_dict()['linear.weight'].data  # Extracting the weights from the model
    w_min = W.min().item()  # Finding minimum weight value
    w_max = W.max().item()  # Finding maximum weight value
    fig, axes = plt.subplots(2, 5)  # Creating a 2x5 grid for displaying weights
    fig.subplots_adjust(hspace=0.01, wspace=0.1)  # Adjusting the spacing between subplots
    for i, ax in enumerate(axes.flat):
        if i < 10:
            ax.set_xlabel("class: {0}".format(i))  # Label each subplot with class number
            ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')  # Displaying weight as an image
            ax.set_xticks([])  # Remove x-ticks
            ax.set_yticks([])  # Remove y-ticks

    plt.show()  # Display the plot


# Function to display a sample image and its label
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28,28), cmap='gray')  # Displaying image
    plt.title('y= ' + str(data_sample[1]))  # Displaying the corresponding label
    plt.show()  # Showing the plot

# Loading MNIST training and validation datasets
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transform.ToTensor())
print("Print the training dataset:: \n", train_dataset)

validation_dataset = dsets.MNIST(root='./data', download=True, transform=transform.ToTensor())
print("Print the validation dataset:\n", validation_dataset)

print("The label: ", train_dataset[3][1])  # Displaying the label of the 3rd sample

# Defining a simple neural network model (SoftMax)
class SoftMax(nn.Module):
    def __init__(self, input_size, output_size):
        super(SoftMax, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # Fully connected layer

    def forward(self, x):
        z = self.linear(x)  # Linear transformation of input
        return z

# Defining input and output sizes
input_dim = 28 * 28  # Flattening the 28x28 image
output_dim = 10      # Output classes (0-9 digits)
model = SoftMax(input_dim, output_dim)
print("Print the model: \n", model)

# Display the weight and bias sizes of the model
print('W: ', list(model.parameters())[0].size())  # Weight size
print('B: ', list(model.parameters())[1].size())  # Bias size

PlotParameters(model)  # Plot the learned weights

# Preprocessing the first image in the dataset
X = train_dataset[0][0]
print(X.shape)  # Shape before flattening
X = X.view(-1, 28 * 28)  # Flattening the image
print(X.shape)  # Shape after flattening
model(X)  # Passing the image through the model

# Initializing optimizer and loss function
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification

# Preparing data loaders for batching
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

# Sample output for the first image in the dataset
model_output = model(X)
actual = torch.tensor([train_dataset[0][1]])  # The actual label of the image

show_data(train_dataset[0])  # Show the image
print("Output:  ", model_output)  # Output from the model (logits)
print("Actual", actual)  # The true label
criterion(model_output, actual)  # Compute the loss

# Applying Softmax to get class probabilities
softmax = nn.Softmax(dim=1)
probability = softmax(model_output)
print(probability)  # Predicted probabilities

# Calculating log loss
print(-1 * torch.log(probability[0][actual]))

# Number of epochs for training
n_epochs = 10
loss_list = []  # List to store loss values over epochs
accuracy_list = []  # List to store accuracy values over epochs
N_test = len(validation_dataset)  # Total number of validation samples

# Function to train the model
def train_model(n_epochs):
    for epoch in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()  # Zeroing out previous gradients
            z = model(x.view(-1, 28 * 28))  # Passing the input through the model
            loss = criterion(z, y)  # Calculating the loss
            loss.backward()  # Backpropagating the gradients
            optimizer.step()  # Updating weights based on gradients

        correct = 0
        for x_test, y_test in validation_loader:
            z = model(x_test.view(-1, 28 * 28))  # Making predictions on validation data
            _, yhat = torch.max(z.data, 1)  # Finding the class with highest probability
            correct += (yhat == y_test).sum().item()  # Counting correct predictions

        accuracy = correct / N_test  # Calculating accuracy
        loss_list.append(loss.data)  # Storing loss for each epoch
        accuracy_list.append(accuracy)  # Storing accuracy for each epoch


train_model(n_epochs)  # Training the model

# Plotting the loss and accuracy
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list, color=color)  # Plotting the loss
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('total loss', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()  # Create a second y-axis for accuracy
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.plot(accuracy_list, color=color)  # Plotting the accuracy
ax2.tick_params(axis='y', color=color)

fig.tight_layout()  # Ensure the layout is adjusted properly
plt.show()  # Display the plot

PlotParameters(model)  # Plot learned weights after training

# Plotting misclassified samples
Softmax_fn = nn.Softmax(dim=-1)  # Softmax function for getting probabilities
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))  # Reshaping image and passing it through the model
    _, yhat = torch.max(z, 1)  # Predicted class
    if yhat != y:  # Check for misclassification
        show_data((x, y))  # Show misclassified sample
        plt.show()
        print("yhat:", yhat)  # Predicted class
        print("probability of class ", torch.max(Softmax_fn(z)).item())  # Predicted class probability
        count += 1
    if count >= 5:  # Show only 5 misclassified samples
        break

# Plotting correctly classified samples
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))  # Reshaping image and passing it through the model
    _, yhat = torch.max(z, 1)  # Predicted class
    if yhat == y:  # Check for correct classification
        show_data((x, y))  # Show correctly classified sample
        plt.show()
        print("yhat:", yhat)  # Predicted class
        print("probability of class ", torch.max(Softmax_fn(z)).item())  # Predicted class probability
        count += 1
    if count >= 5:  # Show only 5 correctly classified samples
        break
