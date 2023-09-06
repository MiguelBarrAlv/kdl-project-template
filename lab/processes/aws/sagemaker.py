import mlflow.sagemaker
import os

from dotenv import load_dotenv


def deploy_model_to_sagemaker(
    model_uri, 
    app_name, 
    image_uri, 
    region_name, 
    aws_sagemaker_role_arn) -> None:
    """
    Deployment model using MLFlow interface.
        :param model_uri: Model uri in MLflow to deploy.
        :param app_name: Application name in SageMaker.
        :param image_uri: Docker image URI in ECR.
        :param region_name: AWS region to deploy. By default, 'eu-north-1'.
    """
    # Get the deployment client
    target_uri = f"sagemaker:/{region_name}"
    client = mlflow.deployments.get_deploy_client(target_uri)
    # Define the deployment configuration
    config = {
        "execution_role_arn": aws_sagemaker_role_arn, 
        "bucket_name": "igz-aws-kdl-training",
        "image_url": image_uri,
        "region_name": region_name,
        "instance_type": "ml.m5.4xlarge",
        "instance_count": 1,
        "synchronous": True,
        "timeout_seconds": 30000,
    }

    # Check if the deployment already exists
    try:
        client.get_deployment(name=app_name)
        deployment_exists = True
    except mlflow.exceptions.MlflowException as e:
        deployment_exists = False

    # Deploy or replace the model
    if deployment_exists:
        client.delete_deployment(name=app_name)
        
    client.create_deployment(name=app_name, model_uri=model_uri, flavor="python_function", config=config)
