import pandas as pd
import numpy as np
import torch
import argparse

from pandas import DataFrame, Series
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Tuple, Union


RANDOM_STATE = 42

def load_cancer_data(return_X_y: bool = True) -> Tuple[DataFrame, Series]:
    """
    Loads breast cancer data as pandas DataFrame (features) and Series (target)
    """
    X, y = load_breast_cancer(return_X_y=True, as_frame=True) 
    return X, y

def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    """
    Splits the data into train/val/test sets
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_csv_and_split(as_type: str) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    """
    Load train/val/test files para X y y from Azure Blob Storage.

    Args:
        datastore_manager: La instancia de AzureDatastoreManager que tiene los detalles de Azure Blob Storage.
        as_type: (str) tipo de outputs; uno de 'array' (devuelve como numpy ndarray)
            o 'tensor' (devuelve como pytorch tensor)

    Returns:
        (tuple) de numpy arrays o torch tensors para
            X_train, X_val, X_test, y_train, y_val, y_test
    """
    imgs, y = load_cancer_data()
    
    # Split into train/test/val
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X=imgs, y=y)

    if as_type == "array":
        return X_train, X_val, X_test, y_train, y_val, y_test

    elif as_type == "tensor":
        # pylint: disable=not-callable
        X_train = torch.Tensor(X_train).float()
        y_train = torch.Tensor(y_train).float()
        X_val = torch.Tensor(X_val).float()
        y_val = torch.Tensor(y_val).float()
        X_test = torch.Tensor(X_test).float()
        y_test = torch.Tensor(y_test).float()

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        raise ValueError(
            "Please specify as_type argument as one of 'array' or 'tensor'"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data and split it")
    
    parser.add_argument("--X_train_data", type=str, required=True)
    parser.add_argument("--X_val_data", type=str, required=True)
    parser.add_argument("--X_test_data", type=str, required=True)
    parser.add_argument("--y_train_data", type=str, required=True)
    parser.add_argument("--y_val_data", type=str, required=True)
    parser.add_argument("--y_test_data", type=str, required=True)
    
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_csv_and_split("array")

    print("X_train_data path_saving:", args.X_train_data)
    np.save(args.X_train_data + ".npy", X_train)
    np.save(args.X_val_data + ".npy", X_val)
    np.save(args.X_test_data + ".npy", X_test)
    np.save(args.y_train_data + ".npy", y_train)
    np.save(args.y_val_data + ".npy", y_val)
    np.save(args.y_test_data + ".npy", y_test)

