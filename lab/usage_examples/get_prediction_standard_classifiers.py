import boto3
import configparser
import json
import os

from dotenv import load_dotenv
from lab.processes.prepare_data.cancer_data import load_data_splits

def predict_with_sagemaker(endpoint_name, sample_data):
    """
    Makes a prediction using a SageMaker endpoint with provided sample data.
    
    Args:
    - endpoint_name (str): Name of the SageMaker endpoint.
    - sample_data (list): Data to predict.
    
    Returns:
    - str: Prediction result from the SageMaker endpoint.
    """
    load_dotenv()
    
    client = boto3.client('sagemaker-runtime',
                          region_name='eu-north-1',
                          aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    try:
        json_payload = json.dumps({'instances': sample_data})
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json_payload,
            ContentType='application/json'
        )
        result_body = response['Body'].read().decode('utf-8')
        prediction = json.loads(result_body)

        return prediction
    
    except client.exceptions.ClientError as e:
        print(f"Endpoint error: {e.response['Error']['Message']}")
        
        return None


# Load training and validation data
PATH_CONFIG = os.getenv("PATH_CONFIG")
config = configparser.ConfigParser()
config.read(str(PATH_CONFIG))

dir_processed = config["paths"]["dir_processed"]
X_train, X_val, _, y_train, y_val, _ = load_data_splits(dir_processed=dir_processed, as_type="array")

sample_data = [X_train[0].tolist()]
prediction = predict_with_sagemaker("mlflow-pyfunc", sample_data)
print(prediction)
