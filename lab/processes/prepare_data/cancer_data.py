"""
Functions for preparing the breast cancer dataset for training and validating ML algorithms
"""
import numpy as np
import os
import pandas as pd
import torch

from azureml.core import Workspace, Datastore
from dotenv import load_dotenv
from pandas import DataFrame, Series
from pathlib import Path
from lab.processes.azure.storage import AzureDatastoreManager
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union
from torch.utils.data import DataLoader
from lib.pytorch import create_dataloader

RANDOM_STATE = 42


def load_cancer_data() -> Tuple[DataFrame, Series]:
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


def prepare_cancer_data(dir_output: str) -> None:
    # NOTE: Refactor function for Azure naming conventions
    """
    Conducts a series of steps necessary to prepare the digit data for training and validation:
    - Loads digit image data from sklearn
    - Splits the data into train, val, and test sets
    - Applies transformations as defined in transform_data
    - Saves output tensors to the destination path provided

    Args:
        dir_output: (str) destination filepath, must be str not PosixPath

    Returns:
        (None)
    """
    load_dotenv() 
    Path(dir_output).mkdir(exist_ok=True)

    # Load digit data
    imgs, y = load_cancer_data()

    # Split into train/test/val
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X=imgs, y=y)

    # Normalize input features:
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    # print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape, "X_val shape: ", X_val.shape, "y_val shape: ",y_val.shape)
    # print(f"X_train, {X_train[:5]}, {y_train[:5]}", "X_val shape: ", X_val.shape, "y_val shape: ",y_val[:5])
    # Save processed data
    np.save(str(Path(dir_output) / "X_train.npy"), X_train.to_numpy())
    np.save(str(Path(dir_output) / "y_train.npy"), y_train.to_numpy())
    np.save(str(Path(dir_output) / "X_val.npy"), X_val.to_numpy())
    np.save(str(Path(dir_output) / "y_val.npy"), y_val.to_numpy())
    np.save(str(Path(dir_output) / "X_test.npy"), X_test.to_numpy())
    np.save(str(Path(dir_output) / "y_test.npy"), y_test.to_numpy())

    # Azure ML
    azure_data_connection = AzureDatastoreManager()
    
    azure_data_connection.upload_data(dir_output)


def load_data_splits_as_dataloader(
    dir_processed: str, batch_size: int, n_workers: int
) -> Tuple[DataLoader]:
    """
    Loads data tensors saved in processed data directory and returns as dataloaders.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_splits(
        dir_processed, as_type="tensor"
    )

    # Convert tensors to dataloaders
    dataloader_args = dict(batch_size=batch_size, num_workers=n_workers, shuffle=True)

    train_loader = create_dataloader(X_train, y_train, dataloader_args)
    val_loader = create_dataloader(X_val, y_val, dataloader_args)
    test_loader = create_dataloader(X_test, y_test, dataloader_args)

    return train_loader, val_loader, test_loader


async def load_data_splits(datastore_manager: AzureDatastoreManager, as_type: str) -> Tuple[Union[np.ndarray, torch.Tensor]]:
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
    # NOTE: Refactor function for Azure naming conventions
    X_train = await datastore_manager.load_data_from_blob("X_train.npy")
    y_train = await datastore_manager.load_data_from_blob("y_train.npy")
    X_val = await datastore_manager.load_data_from_blob("X_val.npy")
    y_val = await datastore_manager.load_data_from_blob("y_val.npy")
    X_test = await datastore_manager.load_data_from_blob("X_test.npy")
    y_test = await datastore_manager.load_data_from_blob("y_test.npy")

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