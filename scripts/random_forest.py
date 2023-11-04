import torch
from torch.utils.data import Dataset, random_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Custom dataset class
class ClassificationDataset(Dataset):
    def __init__(self, input, algos):
        self.input = input
        self.algos = algos

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx: int):
        return self.input[idx], self.algos[idx]
        
def random_forest(data, path):

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

    # Access X and y tensors from train and validation datasets
    # (I am doing it like this because train_test_split isn't allowed to be used in this practical)
    X_train, y_train = train_dataset[:][0], train_dataset[:][1]  # train set
    X_val, y_val = val_dataset[:][0], val_dataset[:][1]  # Validation set

    # Normalize the data
    train_mean = torch.mean(X_train, dim=0)
    train_std = torch.std(X_train, dim=0)
    train_std[train_std == 0] = 1e-7  # add small constant to avoid division by zero

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std  # avoids data leakage (by using the values from train set)

    # Perform PCA on the training data
    pca = PCA(n_components=148)  # Number of PCA components
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    # Train the model
    model = RandomForestClassifier(n_estimators=39, max_depth=68, random_state=50)
    model.fit(X_train_pca, y_train.argmin(axis=1))

    # Predict on the validation set
    y_pred = model.predict(X_val_pca)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_val.argmin(axis=1), y_pred)

    print("Accuracy:", accuracy)

#standalone execution
#if __name__ == '__main__':
#    random_forest()