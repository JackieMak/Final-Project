"""
Implement a 3D CNN model by PyTorch, and make it become a classification model.

Links of access to "Voxel_Result" folder that used in this project:
https://drive.google.com/drive/folders/17hyHuH7MRxylh7_avH8xmlJXO6E8B0-R?usp=sharing
"""

# Importing the libraries
import numpy as np
from sklearn.metrics import f1_score

# PyTorch's libraries and modules
import torch as t
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD

# Import function from others python file
from load_file import LoadFile, load_npy
from permute import Permute
from plot import plot_graph

path_training = "./Voxel_Result/Training_set"
path_validation = "./Voxel_Result/Validation_set"
path_testing = "./Voxel_Result/Testing_set"

load_file = LoadFile(path_training, path_validation, path_testing)

# Get all the files in the folder
X_train_lst, y_train_lst = load_file.get_train_coor(), load_file.get_train_confi()
X_val_lst, y_val_lst = load_file.get_val_coor(), load_file.get_val_confi()
X_test_lst, y_test_lst = load_file.get_test_coor(), load_file.get_test_confi()

X_train_sample, y_train_sample = np.ndarray((0, 20, 20, 20, 4)), np.ndarray((0,))
X_val_sample, y_val_sample = np.ndarray((0, 20, 20, 20, 4)), np.ndarray((0,))
X_test_sample, y_test_sample = np.ndarray((0, 20, 20, 20, 4)), np.ndarray((0,))

for file in X_train_lst:
    training = load_npy(file)
    X_train_sample = np.concatenate((X_train_sample, training), axis=0)

for file in y_train_lst:
    training = load_npy(file)
    y_train_sample = np.concatenate((y_train_sample, training), axis=0)

for file in X_val_lst:
    training = load_npy(file)
    X_val_sample = np.concatenate((X_val_sample, training), axis=0)

for file in y_val_lst:
    training = load_npy(file)
    y_val_sample = np.concatenate((y_val_sample, training), axis=0)

for file in X_test_lst:
    read_file = load_npy(file)
    X_test_sample = np.concatenate((X_test_sample, read_file), axis=0)

for file in y_test_lst:
    read_file = load_npy(file)
    y_test_sample = np.concatenate((y_test_sample, read_file), axis=0)

# Permute the tensor with shape (n, 20, 20, 20, 4) to (n , 4, 20, 20, 20)
X_train = Permute(X_train_sample)
X_val = Permute(X_val_sample)
X_test = Permute(X_test_sample)

train_x = X_train.permute()

train_y = t.from_numpy(y_train_sample).float()
train_y = train_y.unsqueeze(1)
train_y = train_y > 50
train_y = train_y.float()

val_x = X_val.permute()

val_y = t.from_numpy(y_val_sample).float()
val_y = val_y.unsqueeze(1)
val_y = val_y > 50
val_y = val_y.float()

test_x = X_test.permute()

test_y = t.from_numpy(y_test_sample).float()
test_y = test_y.unsqueeze(1)
test_y = test_y > 50
test_y = test_y.float()

print(f"train_x: {train_x.shape} train_y: {train_y.shape}")
print(f"test_x: {val_x.shape} test_y: {val_y.shape}")
print(f"test_x: {test_x.shape}test_y: {test_y.shape}")

# Pick a beforehand batch_size that will use for the training
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Pytorch train and validation sets
train = TensorDataset(train_x, train_y)
val = TensorDataset(val_x, val_y)
test = TensorDataset(test_x, test_y)

device = "cuda:0" if t.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

# data loader
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layer = self._conv_layer_set(4, 100)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(100 * 9 ** 3, 1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _conv_layer_set(in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        return conv_layer

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


# Create 3D-CNN model
my_model = CNNModel()
my_model = my_model.to(device)

# Set up Adam Optimizer
optimizer = Adam(my_model.parameters(), learning_rate)
# optimizer = SGD(my_model.parameters(), learning_rate)

# Set up Binary Cross-Entropy loss
loss_fn = nn.BCELoss()
# loss_fn = nn.MSELoss()

train_loss_lst, train_f1, val_loss_lst, val_f1, test_loss_lst, test_f1 = [], [], [], [], [], []

# This is the main training loop
for epoch in range(num_epochs):
    # First loop over the training data and calculate loss as well as collecting predictions/targets to calculate
    # other metrics.
    # Also do the loss.backward() and optimizer.step() to actually improve the model.
    train_loss = 0
    train_predictions, train_targets = [], []
    for input, target in train_loader:
        input = input.to(device)
        target = target.to(device)

        output = my_model(input)

        loss = loss_fn(output, target)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_pred = [round(x) for x in output.detach().flatten().tolist()]
        batch_targets = [round(x) for x in target.detach().flatten().tolist()]

        train_predictions += batch_pred
        train_targets += batch_targets

    # Same idea for the validation and testing dataset. Calculate loss and collect the targets and predictions for later
    # performance metric calculation.
    # However use torch.no_grad() and no updating the model using the optimizer.
    val_loss = 0
    val_predictions, val_targets = [], []
    for input, target in val_loader:
        input = input.to(device)
        target = target.to(device)

        with t.no_grad():
            output = my_model(input)

            loss = loss_fn(output, target)
            val_loss += loss

            batch_pred = [round(x) for x in output.detach().flatten().tolist()]
            batch_targets = [round(x) for x in target.detach().flatten().tolist()]

            val_predictions += batch_pred
            val_targets += batch_targets

    test_loss = 0
    test_predictions, test_targets = [], []
    for input, target in test_loader:
        input = input.to(device)
        target = target.to(device)

        with t.no_grad():
            output = my_model(input)

            loss = loss_fn(output, target)
            test_loss += loss

            batch_pred = [round(x) for x in output.detach().flatten().tolist()]
            batch_targets = [round(x) for x in target.detach().flatten().tolist()]

            test_predictions += batch_pred
            test_targets += batch_targets

    # Average the loss over the number of samples in the training, validation and testing set.
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    test_loss /= len(test_loader)

    train_loss_lst.append(train_loss.detach().cpu().numpy())
    val_loss_lst.append(val_loss.detach().cpu().numpy())
    test_loss_lst.append(test_loss.detach().cpu().numpy())

    # Calculate an F1 score using the collected targets and predictions for the training, validation & testing datasets.
    train_F1 = f1_score(train_targets, train_predictions)
    val_F1 = f1_score(val_targets, val_predictions)
    test_F1 = f1_score(test_targets, test_predictions)

    train_f1.append(train_F1)
    val_f1.append(val_F1)
    test_f1.append(test_F1)

    print(f"epoch {epoch + 1}.")
    print(f" train_loss={train_loss} train_F1={train_F1}")
    print(f" val_loss={val_loss} val_F1={val_F1}")
    print(f" test_loss={test_loss} test_F1={test_F1}")

# For finding out the value of batch_size, learning_rate and num_epochs have the maximum value of F1 score
print(f"\nmax_val_f1={max(val_f1)}")
# After finding those parameters, test for the testing set to see if the testing dataset have the similar maximum
# F1 score compare to validation dataset.
print(f"max_test_f1={max(test_f1)}")

# Plot line graph for the loss and F1 score of training, validation & testing datasets.
plot_graph(num_epochs, train_loss_lst, train_f1, val_loss_lst, val_f1, test_loss_lst, test_f1)
