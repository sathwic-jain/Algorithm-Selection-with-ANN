import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os

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

def loss_plot(losses, part):
    # Plot the losses
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title('Training Loss')
    
    plots_directory = "plots/"
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)
    
    plt.savefig(f"{plots_directory}{part}_loss_plot.png")
      
        
class Classification(nn.Module):
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

# validation
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
            loss = criterion(outputs, y)  # Calculate loss
            total_loss += loss.item()

        total_loss /= len(val_loader)  # Calculate average validation loss
        print()  # Print an empty line for visual separation
        print(f"Validation loss: {total_loss}")  # Print the validation loss

    return total_loss
        
        
def classification_model(data, path):

    # Define the paths to the input files
    instance_file = f'{data}instance-features.txt'
    performance_file = f'{data}performance-data.txt'

    # Load the data from the instance file
    with open(instance_file, 'r') as f:
        instance_data = f.readlines()

    # Convert the data to a tensor
    X_data = torch.tensor([list(map(float, line.strip().split())) for line in instance_data], dtype=torch.float32)

    # Load the data from the performance file
    with open(performance_file, 'r') as f:
        performance_data = f.readlines()

    # Convert the data to a tensor
    y_data = torch.tensor([list(map(float, line.strip().split())) for line in performance_data], dtype=torch.float32)

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

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    X_train, y_train = train_dataset[:][0], train_dataset[:][1]
    X_val, y_val = val_dataset[:][0], val_dataset[:][1]

    # Normalize the data
    train_mean = torch.mean(X_train, dim=0)
    train_std = torch.std(X_train, dim=0)
    train_std[train_std == 0] = 1e-8 #avoid division by zero

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    


    # Reassign the standardized data to the datasets
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    
    # Set the HyperParameters
    input_size = X_train.shape[1]  # Number of columns in the training data represents the number of features
    output_size = y_train.shape[1]  # Number of columns in the target data represents the number of output dimensions
    hidden_layer_size = 120  # Size of the hidden layer in the model
    num_epochs = 100  # Number of times the entire training dataset is passed through the model during training
    batch_size = 250  # Number of samples per batch used for training and evaluation
    learning_rate = 0.01003227325872551  # The rate at which the model learns from the training data
    weight_decay = 3.108035567629965e-05  # A regularization technique that helps prevent overfitting by adding a penalty to the loss function


    # data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create the model
    model = Classification(input_size, output_size, hidden_layer_size)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model
    # train(model, train_loader, optimizer, criterion, num_epochs)
    model.train()  # Set the model to training mode
    patience = 50
    #training will stop if the validation loss doesn't improve for 10 consecutive epochs.
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []  # List to store the training losses

    for epoch in range(num_epochs):
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        val_loss = validate(model, val_loader, criterion)  # Add validation loss computation during training
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement == patience:
            print("Early stopping triggered. Training stopped after {} epochs.".format(epoch + 1))
            break

    loss_plot(train_losses, 'Classification_P2_1')  # Generate the training loss plot
    # Save the model
    model_details = {
        'input_size': input_size,
        'output_size': output_size,
        'hidden_layer_size': 120,
        'model': model.state_dict(),
        'train_mean': train_mean,
        'train_std': train_std,
        'batch_size': batch_size
    }

    #path = '../models/classification_basic.pth'
    torch.save(model_details, path)

#standalone running of the script
#if __name__ == "__main__":
#    classification_model()