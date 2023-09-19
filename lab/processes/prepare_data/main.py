"""
ML pipeline for breast cancer classification
Part 1: Data preparation
"""
import configparser
import os

from lab.processes.prepare_data.cancer_data import upload_dataframe_to_azure_blob
from lab.processes.azure.storage import AzureDatastoreManager

PATH_CONFIG = os.getenv("PATH_CONFIG")
# Azure Blob Storage
# AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')

config = configparser.ConfigParser()
config.read(str(PATH_CONFIG))
DIR_DATA_PROCESSED = config["paths"]["dir_processed"]

# CONTAINER_NAME = config["azure"]["container_name"]
# blob_helper = AzureBlobHelper(AZURE_CONNECTION_STRING)
# blob_helper.ensure_container_exists(CONTAINER_NAME)

if __name__ == "__main__":
    # prepare_cancer_data(dir_output=DIR_DATA_PROCESSED)
    azure_data_connection = AzureDatastoreManager()
    blob_name = 'breast_cancer_data.csv'
    upload_dataframe_to_azure_blob(blob_name, azure_data_connection)
