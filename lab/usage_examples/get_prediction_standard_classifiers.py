import asyncio
import configparser
import json
import os
import ssl
import urllib.request

from lab.processes.azure.storage import AzureDatastoreManager
from lab.processes.prepare_data.cancer_data import load_data_splits


def predict_with_azure(endpoint_url, api_key, model_name, sample_data):
    """
    Makes a prediction using an Azure endpoint with provided sample data.
    
    Args:
    - endpoint_url (str): URL of the Azure endpoint.
    - api_key (str): API key for the Azure endpoint.
    - model_name (str): Name of the model deployment.
    - sample_data (list): Data to predict.
    
    Returns:
    - str: Prediction result from the Azure endpoint.
    """
    # Convert the sample data to the required input format for Azure
    data = {
        "input_data": sample_data
    }

    body = str.encode(json.dumps(data))
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key,
        'azureml-model-deployment': model_name
    }

    req = urllib.request.Request(endpoint_url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result)
    
    except urllib.error.HTTPError as error:
        print(f"Endpoint error with status code: {error.code}")
        print(error.read().decode("utf8", 'ignore'))
        return None


# Cargar configuración desde .env
#load_dotenv()

datastore_manager = AzureDatastoreManager()

X_train, X_val, X_test, y_train, y_val, y_test = asyncio.run(load_data_splits(datastore_manager, "array"))

sample_data = [X_train[0].tolist()]
api_key = ""
model_name = ""
endpoint_url = ""
prediction = predict_with_azure(endpoint_url, api_key, model_name, sample_data)

print(prediction)
