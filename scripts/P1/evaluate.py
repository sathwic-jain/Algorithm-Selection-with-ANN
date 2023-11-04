import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from regression import ClassificationDataset, Regression
import os

def main():
    #usage
    #python evaluate.py --model <model_path> --data <data_path>
    parser = argparse.ArgumentParser(description="Evaluate a trained AS model on a test set")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    
    args = parser.parse_args()

    print(f"\nLoading trained model {args.model} and evaluating it on {args.data}")
    
    # load the given model, make predictions on the given dataset and evaluate the model's performance. Your evaluation should report four evaluation metrics: avg_loss, accuracy, avg_cost, sbs_vbs_gap (as listed below)
    # you should also calculate the average cost of the SBS and the VBS
    avg_loss = np.inf # the average loss value across the given dataset
    accuracy = 0 # classification accuracy 
    avg_cost = np.inf # the average cost of the predicted algorithms on the given dataset
    sbs_vbs_gap = np.inf # the SBS-VBS gap of your model on the given dataset
    sbs_avg_cost = np.inf # the average cost of the SBS on the given dataset 
    vbs_avg_cost = np.inf # the average cost of the VBS on the given dataset
    # YOUR CODE HERE

    # Load the saved model
    model_details = torch.load(f"{args.model}")
    
    input_size = model_details["input_size"]
    output_size = model_details["output_size"]
    hidden_size = model_details["hidden_layer_size"]
    model = Regression(input_size, output_size, hidden_size)
    model.load_state_dict(model_details["model"])
    train_mean = model_details["train_mean"]
    train_std = model_details["train_std"]
    batch_size = model_details["batch_size"]

    # Define the paths to the input files
    instance_file = f'{args.data}instance-features.txt'
    performance_file = f'{args.data}performance-data.txt'

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

    epsilon = 1e-7  # to the standard deviation to avoid division by zero
    X_data = (X_data - train_mean) / (train_std + epsilon) # normalize the data

    # Creating dataset
    dataset = ClassificationDataset(X_data, y_data)
    # Creating data loader
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
   #using the loss function used in training
    criterion = nn.MSELoss()

    #Calculating svs and vbs costs
    
    sbs_avg_cost = y_data.mean(dim=0).min()
    vbs_avg_cost = y_data.min(dim=1).values.mean()

    #Calculating the average loss and accuracy
    avg_loss = 0
    accuracy = 0
    avg_cost = 0
    sbs_vbs_gap = 0
    for i,(inputs, targets) in enumerate(data_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        avg_loss += loss.item()
        avg_cost += outputs.mean(dim=0).min()
        accuracy += (outputs.min(dim=1).indices == targets.min(dim=1).indices).sum().item()
    avg_loss /= len(data_loader)
    accuracy /= len(dataset)
    avg_cost /= len(data_loader)
    sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)
    
    # print results
    print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")
    #print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}", file=open(f"../results/regression_model_result.txt","a"))


    result_file_path = "results/regression_model_result.txt"
    result_directory = os.path.dirname(result_file_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    mode = "a" if os.path.exists(result_file_path) else "w"

    with open(result_file_path, mode) as result_file:
        print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}", file=result_file)
if __name__ == "__main__":
    main()
