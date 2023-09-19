import argparse
import numpy as np

from azureml.core import Run
from manager import MLFlowManager
from train_classifiers import train_classifiers

def main():

    parser = argparse.ArgumentParser(description="Process inputs for training.")
    parser.add_argument("--X_train_data", type=str, required=True)
    parser.add_argument("--X_val_data", type=str, required=True)
    parser.add_argument("--X_test_data", type=str, required=True)
    parser.add_argument("--y_train_data", type=str, required=True)
    parser.add_argument("--y_val_data", type=str, required=True)
    parser.add_argument("--y_test_data", type=str, required=True)
    
    args = parser.parse_args()
    print("X_train_data path:", args.X_train_data)
    X_train = np.load(args.X_train_data + ".npy")
    X_val = np.load(args.X_val_data + ".npy")
    X_test = np.load(args.X_test_data + ".npy")
    y_train = np.load(args.y_train_data + ".npy") 
    y_val = np.load(args.y_val_data + ".npy")
    y_test = np.load(args.y_test_data + ".npy")

    manager_mlflow = MLFlowManager()
    train_classifiers(manager_mlflow, X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()

