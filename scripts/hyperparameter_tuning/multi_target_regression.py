import argparse
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json


# Custom dataset class
class ClassificationDataset(Dataset):
    def __init__(self, input, algos):
        self.input = input
        self.algos = algos

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx: int):
        return self.input[idx], self.algos[idx]


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()
        hl_1_size = int(3*hidden_layer_size/4)
        hl_2_size = int(hidden_layer_size/2)
        hl_3_size = int(hidden_layer_size/4)
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.hl_1 = nn.Linear(hidden_layer_size, hl_1_size) #Epoch [100/100], Loss: 55296086.4430, validation loss: 54070573.526315786
        self.hl_2 = nn.Linear(hl_1_size, hl_2_size) #Epoch [100/100], Loss: 55296086.4430, validation loss: 54070573.526315786)) 
        self.hl_3 = nn.Linear(hl_2_size, hl_3_size) #Epoch [200/200], Loss: 54535805.3441 Validation loss: 52458433.36170213
        self.output_layer = nn.Linear(hl_3_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = self.input_layer(data)
        x = self.relu(x)
        x = self.hl_1(x)
        x = self.relu(x)
        x = self.hl_2(x)
        x = self.relu(x)
        x = self.hl_3(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class ModelLogistic(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()
        hl_1_size = int(2*hidden_layer_size/3)
        hl_2_size = int(hidden_layer_size/3)
        self.input_layer = nn.Linear(input_size, hl_1_size)
        self.hl_1 = nn.Linear(hl_1_size, hl_2_size)
        self.hl_2 = nn.Linear(hl_2_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, data):
        x = self.input_layer(data)
        x = self.sig(x)
        x = self.hl_1(x)
        x = self.sig(x)
        x = self.hl_2(x)
        x = self.sig(x)
        x = self.output_layer(x)
        return x


# train
def train(model, train_loader, optimizer, criterion, num_epochs):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # model performance during training.
        # If used loss instead of train_loss in the loop,
        # It would be overwriting the loss value for each batch
        train_loss /= len(train_loader)

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss))


# validation
def validate(model, val_loader, criterion):
    # Switch to evaluation mode
    model.eval()  # method disables the dropout and batch normalization layers in the model
    with torch.no_grad():
        total_loss = 0.0
        for X, y in val_loader:
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()

        total_loss /= len(val_loader)
        print()
        print(f"Validation loss: {total_loss}")
        return total_loss


if __name__ == "__main__":

    # Define the paths to the input files
    instance_file = '../../data/train/instance-features.txt'
    performance_file = '../../data/train/performance-data.txt'

    # Load the data from the instance file
    with open(instance_file, 'r') as f:
        instance_data = f.readlines()

    # Convert the data to a tensor
    X_data = torch.tensor([list(map(float, line.strip().split())) for line in instance_data])

    # Load the data from the performance file
    with open(performance_file, 'r') as f:
        performance_data = f.readlines()

    # Convert the data to a tensor
    y_data = torch.tensor([list(map(float, line.strip().split())) for line in performance_data])

    # Check for null values
    print("Null values in X_data:", torch.isnan(X_data).any())
    print("Null values in y_data:", torch.isnan(y_data).any())

    # Check for duplicates
    print("Duplicates in X_data:", len(X_data) != len(torch.unique(X_data, dim=0)))
    print("Duplicates in y_data:", len(y_data) != len(torch.unique(y_data, dim=0)))

    dataset = ClassificationDataset(X_data, y_data)  # custom dataset for pytorch

    # Determine the sizes of the training and validation sets
    train_ratio = 0.8  # 80% for training, 20% for validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    torch.manual_seed(50)  # Set the seed for the random number generator
    # split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Access X and y tensors from train and validation datasets
    # (I am doing it like this because train_test_split isn't allowed to be used in this practical)
    X_train, y_train = train_dataset[:][0], train_dataset[:][1]  # train set
    X_val, y_val = val_dataset[:][0], val_dataset[:][1]  # Validation set

    print("Shape after split")
    print("-----------------")
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_val: ", X_val.shape)
    print("y_val: ", y_val.shape)
    print()

    # Normalize the data
    train_mean = torch.mean(X_train, dim=0)
    train_std = torch.std(X_train, dim=0)
    train_std[train_std == 0] = 1e-7  # add small constant to avoid division by zero

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std  # avoids data leakage (by using the values from train set)

    # reassignment
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)

    # HyperParameters
    input_size = X_train.shape[1]  # as columns represent no: of features
    output_size = y_train.shape[1]

    # Define the loss function
    criterion = nn.MSELoss()

    def objective(trial):
        # Tunable HyperParameters
        learning_rate = trial.suggest_float('learning_rate', 1e-2, 5e-2, log=True)  # perform better with a logarithmic search
        num_epochs = trial.suggest_int('num_epochs', 100, 300)
        batch_size = trial.suggest_int('batch_size', 15, 35)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)  # regularization

        # data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # create the model
        model = RegressionModel(input_size, output_size, 120)

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train the model
        train(model, train_loader, optimizer, criterion, num_epochs)

        # Evaluate the model
        validation_loss = validate(model, val_loader, criterion)

        return validation_loss

    # create study object
    study = optuna.create_study(direction='minimize')  # minimise as total_loss is used for evaluation in validation

    # run optimization
    study.optimize(objective, n_trials=30)

    # print best HyperParameters
    print(f"Best HyperParameters: {study.best_params}")
    # Save best_params to a txt file
    file_path = 'hyperparameters.txt'
    with open(file_path, 'w') as f:
        json.dump(study.best_params)

    print(f"Hyperparameters saved to {file_path}")
    print(f"\nHyperParameter Tuning finished")


