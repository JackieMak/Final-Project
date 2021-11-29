# Importing the libraries
import numpy as np

# For reading and displaying images
# import matplotlib.pyplot as plt

# PyTorch libraries and modules
import torch as t
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import SGD

# Import load_npy function from load_file.py
from load_file import LoadFile, load_npy
from permute import Permute

path_training = "./Voxel_Result/Training_set"
path_validation = "./Voxel_Result/Validation_set"
path_testing = "./Voxel_Result/Testing_set"

load_file = LoadFile(path_training, path_validation, path_testing)

# Get all the files in the folder
X_train_lst = load_file.get_train_coor()
y_train_lst = load_file.get_train_confi()
X_test_lst = load_file.get_val_coor()
y_test_lst = load_file.get_val_confi()

X_train_sample = np.ndarray((0, 20, 20, 20, 4))
y_train_sample = np.ndarray((0,))
X_test_sample = np.ndarray((0, 20, 20, 20, 4))
y_test_sample = np.ndarray((0,))

for file in X_train_lst:
    training = load_npy(file)
    X_train_sample = np.concatenate((X_train_sample, training), axis=0)

for file in y_train_lst:
    training = load_npy(file)
    y_train_sample = np.concatenate((y_train_sample, training), axis=0)

for file in X_test_lst:
    training = load_npy(file)
    X_test_sample = np.concatenate((X_test_sample, training), axis=0)

for file in y_test_lst:
    training = load_npy(file)
    y_test_sample = np.concatenate((y_test_sample, training), axis=0)

# Permute the tensor with shape (n, 20, 20, 20, 4) to (n , 4, 20, 20, 20)
X_train = Permute(X_train_sample)
X_test = Permute(X_test_sample)

train_x = X_train.permute()
train_y = t.from_numpy(y_train_sample).float()
test_x = X_test.permute()
test_y = t.from_numpy(y_test_sample).float()

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# Pick a before-hand batch_size that will use for the training
batch_size = 100

# Pytorch train and test sets
train = t.utils.data.TensorDataset(train_x, train_y)
test = t.utils.data.TensorDataset(test_x, test_y)

# data loader
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layer1 = nn.Conv3d(4, 100, kernel_size=(3, 3, 3), padding=0)
        self.conv_layer2 = self._conv_layer_set(100, 200)
        self.conv_layer3 = self._conv_layer_set(200, 400)
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 ** 3 * 400, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.batch = nn.BatchNorm1d(1000)
        self.drop = nn.Dropout(p=0.15)

    @staticmethod
    def _conv_layer_set(in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)  # torch.Size([100, 10800])
        # out = self.flatten(out)
        # print(out.shape)
        out = self.fc1(out)
        # print(out.shape)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(100)

        return out


# Definition of hyper-parameters
n_iters = 4500
num_epochs = n_iters / (len(train_x) / batch_size)
num_epochs = int(num_epochs)

# Create 3D-CNN model
my_model = CNNModel()
# print(my_model)

# Set up SGD Optimizer
learning_rate = 0.001
optimizer = SGD(my_model.parameters(), learning_rate)

# Set up mean-squared error
loss_fn = nn.MSELoss()

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(100, 4, 20, 20, 20))  #
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = my_model(train)
        # Calculate softmax and ross entropy loss
        loss = loss_fn(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()

        count += 1
        if count % 50 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for _images, _labels in test_loader:
                test = Variable(_images.view(100, 4, 20, 20, 20))  #
                # Forward propagation
                outputs = my_model(test)

                # Get predictions from the maximum value
                predicted = t.max(outputs.data, 1)[1]

                # Total number of labels
                total += len(_labels)
                correct += (predicted == _labels).sum()

            accuracy = 100 * correct / float(total)

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

    print(f"{epoch+1}/{num_epochs}")
