import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import importlib.util

# Specify the path to the module
module_path = '../P2_2/P2_2_cost.py'

# Load the module dynamically
spec = importlib.util.spec_from_file_location('P2_2_cost', module_path)
P2_2_cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P2_2_cost)

# # Now you can use the functions/classes from P2_2_cost module
# P2_2_cost.custom_loss()


# Custom dataset class for classification
class ClassificationDataset(Dataset):
    def __init__(self, input, algos):
        """
        Initialize the ClassificationDataset.

        Args:
            input (torch.Tensor): Input data tensor.
            algos (torch.Tensor): Target algorithm tensor.
        """
        self.input = input
        self.algos = algos

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.input.shape[0]

    def __getitem__(self, idx: int):
        """
        Get an item from the dataset at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input data and target algorithm.
        """
        return self.input[idx], self.algos[idx]



class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()
        
        # Define the sizes of hidden layers
        hl_1_size = int(3 * hidden_layer_size / 4)
        hl_2_size = int(hidden_layer_size / 2)
        hl_3_size = int(hidden_layer_size / 4)
        
        # Define the layers of the model
        self.input_layer = nn.Linear(input_size, hidden_layer_size)  # Input layer
        self.hl_1 = nn.Linear(hidden_layer_size, hl_1_size)  # Hidden layer 1
        self.hl_2 = nn.Linear(hl_1_size, hl_2_size)  # Hidden layer 2
        self.hl_3 = nn.Linear(hl_2_size, hl_3_size)  # Hidden layer 3
        self.output_layer = nn.Linear(hl_3_size, output_size)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        
    def forward(self, data):
        # Feedforward propagation through the model
        
        x = self.input_layer(data)  # Pass input through input layer
        x = self.relu(x)  # Apply ReLU activation function
        
        x = self.hl_1(x)  # Pass input through hidden layer 1
        x = self.relu(x)  # Apply ReLU activation function
        
        x = self.hl_2(x)  # Pass input through hidden layer 2
        x = self.relu(x)  # Apply ReLU activation function
        
        x = self.hl_3(x)  # Pass input through hidden layer 3
        x = self.relu(x)  # Apply ReLU activation function
        
        x = self.output_layer(x)  # Pass input through output layer
        
        return x



def validate(model, val_loader, criterion):
    """
    Perform validation on the given model using the validation data.

    Args:
        model (nn.Module): The trained model to be validated.
        val_loader (DataLoader): The data loader for the validation dataset.
        criterion: The loss criterion used for evaluation.

    Returns:
        float: The average validation loss.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_loss = 0.0
        for X, y in val_loader:
            outputs = model(X)  # Forward pass
            loss = P2_2_cost.custom_loss(outputs, y)  # Calculate loss
            total_loss += loss.item()

        total_loss /= len(val_loader)  # Calculate average validation loss
        print()  # Print an empty line for visual separation
        print(f"Validation loss: {total_loss}")  # Print the validation loss

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

    dataset = ClassificationDataset(X_data, y_data)  # Custom dataset for PyTorch

    # Determine the sizes of the training and validation sets
    train_ratio = 0.8  # 80% for training, 20% for validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    torch.manual_seed(50)  # Set the seed for the random number generator

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    X_train, y_train = train_dataset[:][0], train_dataset[:][1]
    X_val, y_val = val_dataset[:][0], val_dataset[:][1]

    # Standardize the data
    train_mean = torch.mean(X_train, dim=0)
    train_std = torch.std(X_train, dim=0)
    train_std[train_std == 0] = 1e-8

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    # Reassign the standardized data to the datasets
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)

    # Define the hyperparameters
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    # Define the loss function
    criterion = nn.MSELoss()

    def objective(trial):
        # Tunable HyperParameters
        learning_rate = trial.suggest_float('learning_rate', 1e-2, 5e-2, log=True)
        num_epochs = trial.suggest_int('num_epochs', 100, 300)
        batch_size = trial.suggest_int('batch_size', 200, 300)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

        # data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # create the model
        model = RegressionModel(input_size, output_size, 120)

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train the model
        model.train()
        for epoch in range(num_epochs):
            train_loss = 0.0
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = model(X)
                loss = P2_2_cost.custom_loss(outputs, y)
                #loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")       

        # Evaluate the model
        validation_loss = validate(model, val_loader, criterion)

        return validation_loss

    # create study object
    study = optuna.create_study(direction='minimize')

    # run optimization
    study.optimize(objective, n_trials=30)

    # print best HyperParameters
    print(f"Best HyperParameters: {study.best_params}")
    # Save best_params to a txt file
    file_path = 'hyperparameters_cls_adv.txt'
    with open(file_path, 'w') as f:
        json.dump(study.best_params,f)

    print(f"Hyperparameters saved to {file_path}")
    print(f"\nHyperParameter Tuning finished")


