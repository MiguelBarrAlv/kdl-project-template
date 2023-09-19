import argparse
import pandas as pd

from sklearn.datasets import load_breast_cancer
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from lab.processes.azure.storage import AzureDatastoreManager
from io import BytesIO


def upload_dataframe_to_azure_blob(blob_name: str, azure_data_connection) -> None:
    """
    Loads the breast cancer dataset, unifies it, and uploads it directly 
    to Azure Blob Storage without saving it locally.
    """
    data = load_breast_cancer(return_X_y=False, as_frame=True)
    df = data['data']
    df['target'] = data['target']

    data_io = dataframe_to_bytesio(df)
    azure_data_connection = AzureDatastoreManager()

    try:
        azure_data_connection.upload_data_from_stream(data_io, blob_name)
        print(f"Data uploaded to: {azure_data_connection.blob_datastore_name}/{azure_data_connection.container_name}")
    except ResourceExistsError:
        print(f"Blob already exists in Azure Blob Storage with path {azure_data_connection.blob_datastore_name}/{azure_data_connection.container_name}")


def dataframe_to_bytesio(df: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blob_name", type=str, default='breast_cancer_data.csv')
    args = parser.parse_args()
    azure_data_connection = AzureDatastoreManager()
    blob_name = 'breast_cancer_data.csv'
    upload_dataframe_to_azure_blob(blob_name, azure_data_connection)
