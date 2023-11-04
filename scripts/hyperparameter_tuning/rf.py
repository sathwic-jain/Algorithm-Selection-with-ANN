import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json


if __name__ == "__main__":
    # Load the data
    X_data = np.loadtxt('../../data/train/instance-features.txt')
    Y_data = np.loadtxt('../../data/train/performance-data.txt')

    def objective(trial):
        # Define tunable HyperParameters
        pca_n = trial.suggest_int('pca_components', 100, 150)  # Number of PCA components
        depth = trial.suggest_int('depth', 10, 100)  # Maximum depth of the random forest
        n_estimators = trial.suggest_int('n_estimators', 10, 100)  # Number of estimators (trees) in the random forest

        # Split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=50)

        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Apply PCA
        pca = PCA(n_components=pca_n)  # Retain the specified number of PCA components
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        # Train the model
        model = RandomForestClassifier(class_weight="balanced", max_depth=depth, random_state=0, n_estimators=n_estimators)
        model.fit(X_train_pca, y_train.argmax(axis=1))  # Use argmax instead of argmin

        # Predict on the validation set
        y_pred = model.predict(X_val_pca)

        # Calculate accuracy
        accuracy = accuracy_score(y_val.argmax(axis=1), y_pred)  # Use argmax instead of argmin

        return accuracy

    # Create an Optuna study object to optimize the objective function
    study = optuna.create_study(direction='maximize')  # Maximize accuracy as the evaluation metric

    # Run the optimization
    study.optimize(objective, n_trials=500)

    # Print the best HyperParameters
    print(f"Best HyperParameters: {study.best_params}")

    # Save the best_params to a txt file
    file_path = 'random_forest.txt'
    with open(file_path, 'w') as f:
        json.dump(study.best_params, f)

    print(f"\nHyperParameter Tuning finished")
