"""
This script provides an example of building a binary neural
network for classifying glass identification dataset on
http://archive.ics.uci.edu/ml/datasets/Glass+Identification
"""

# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

"""
Step 1: Load data and pre-process data
"""
# load all data
data = pd.read_csv('glass.csv',  header=None)
# drop first column as it is identifier
data.drop(data.columns[0], axis=1, inplace=True)

# try shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# split data into 2 classes to perform binary classification
# by replacing class values in the range of 1-4 (window glass)
# with 1, and 5-7 (non-window glass) with 0
data[data.shape[1]] = (data[data.shape[1]] < 5).astype(int)

# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]

n_features = train_data.shape[1] - 1

# split training data into input and target
# the first 9 columns are features, the last one is target
train_input = train_data.iloc[:, :n_features]
train_target = train_data.iloc[:, n_features]

# normalise training data by columns
#for column in train_input:
#    train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max()
#    - x.min()))
    # train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# split training data into input and target
# the first 9 columns are features, the last one is target
test_input = test_data.iloc[:, :n_features]
test_target = test_data.iloc[:, n_features]

# normalise testing input data by columns
#for column in test_input:
#    test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() -
#x.min()))
    # test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# create Tensors to hold inputs and outputs
X = torch.Tensor(train_input.values).float()
Y = torch.Tensor(train_target.values).long()

"""
Step 2: Define a neural network 

Here we build a neural network with one hidden layer.
    input layer: 9 neurons, representing the features of Glass
    hidden layer: 10 neurons, using Sigmoid as activation function
    output layer: 2 neurons, representing the type of glass

The network will be trained with Stochastic Gradient Descent (SGD) as an
optimiser, that will hold the current state and will update the parameters
based on the computed gradients.

Its performance will be evaluated using cross-entropy.
"""

# define the number of inputs, classes, training epochs, and learning rate
input_neurons = n_features
hidden_neurons = 10
output_neurons = 2
learning_rate = 0.01
num_epochs = 500


# define a customised neural network structure
class TwoLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        # get hidden layer input
        h_input = self.hidden(x)
        # define activation function for hidden layer
        h_output = torch.sigmoid(h_input)
        # get output layer output
        y_pred = self.out(h_output)

        return y_pred

# define a neural network using the customised structure
net = TwoLayerNet(input_neurons, hidden_neurons, output_neurons)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# train a neural network
for epoch in range(num_epochs):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)

    # Compute loss
    loss = loss_func(Y_pred, Y)
    all_losses.append(loss.item())

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(Y_pred, 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct)/total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss

# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(all_losses)
# plt.show()

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every glass (rows)
which class the network guesses (columns).

"""

confusion = torch.zeros(output_neurons, output_neurons)

Y_pred = net(X)

_, predicted = torch.max(Y_pred, 1)

for i in range(train_data.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)


"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""

# create Tensors to hold inputs and outputs
X_test = torch.Tensor(test_input.values).float()
Y_test = torch.Tensor(test_target.values).long()

# test the neural network using testing data
# It is actually performing a forward pass computation of predicted y
# by passing x to the model.
# Here, Y_pred_test contains three columns, where the index of the
# max column indicates the class of the instance
Y_pred_test = net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every iris flower (rows)
which class the network guesses (columns).

"""

confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)
