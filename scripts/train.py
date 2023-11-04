import argparse
from P1.regression import regression_model
from P2_1.P2_1_classification import classification_model
from P2_2.P2_2_cost import classification_cost_model
from random_forest import random_forest

def main():
    parser = argparse.ArgumentParser(description="Train an AS model and save it to file")
    parser.add_argument("--model-type", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    parser.add_argument("--save", type=str, required=True, help="Save the trained model (and any related info) to a .pt file")
    
    args = parser.parse_args()

    print(f"\nTraining a {args.model_type} model on {args.data}, and save it to {args.save}")
    
    if(args.model_type == 'regresion_nn'):
        regression_model(args.data, args.save)     
        print()
    
    if(args.model_type == 'classification_nn'):
        classification_model(args.data, args.save)
        print()
        
    if(args.model_type == 'classification_nn_cost'):
        classification_cost_model(args.data, args.save)
        print()        
        
    if(args.model_type == 'random_forest'):
        random_forest(args.data, args.save)  
        print()

    # print results
    print(f"\nTraining finished")


if __name__ == "__main__":
    main()
