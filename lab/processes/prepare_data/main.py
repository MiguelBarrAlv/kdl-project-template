"""
ML pipeline for breast cancer classification
Part 1: Data preparation
"""

import configparser
import os

from lab.processes.prepare_data.cancer_data import prepare_cancer_data

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
    prepare_cancer_data(dir_output=DIR_DATA_PROCESSED)
