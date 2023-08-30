import mlflow.sagemaker


# def deploy_model_to_sagemaker(model_uri, app_name, image_uri, region_name='eu-north-1'):
#     # """
#     # Desplegar el modelo en SageMaker.
#     # """
#     # repository_name = image_url.split("/")[-1].split(":")[0]

#     # # Subir la imagen Docker a ECR
#     # mfs.push_image_to_ecr(image=repository_name)
    
#     # Crear un modelo en SageMaker
#     # mfs.deploy(
#     #     model_uri=model_uri,
#     #     region_name=region_name,
#     #     image_url=image_url,
#     #     execution_role_arn="arn:aws:iam::688013747199:user/kdl-template", # NOTE: Hardcoded for now
#     #     instance_type="ml.t2.medium"  # Tipo de instancia gratuito en SageMaker
#     # )

#     mlflow.sagemaker.deploy(
#     mode='create',
#     app_name='kdl-endpoint',
#     model_uri=model_uri,
#     image_url=image_uri,
#     instance_type='ml.m5.xlarge',
#     instance_count=1,
#     region_name=region
# )

import mlflow.deployments

def deploy_model_to_sagemaker(model_uri, app_name, image_uri, region_name='eu-north-1'):
    """
    Deployment model using MLFlow interface.
        :param model_uri: Model uri in MLflow to deploy.
        :param app_name: Application name in SageMaker.
        :param image_uri: Docker image URI in ECR.
        :param region_name: AWS region to deploy. By default, 'eu-north-1'.
        :param execution_role_arn: SageMaker execution role ARN.
    """
    # Get the deployment client
    target_uri = f"sagemaker:/{region_name}"
    client = mlflow.deployments.get_deploy_client(target_uri)
    # Configuration for the deployment
    config = {
        "execution_role_arn": "arn:aws:iam::688013747199:role/kdl-sagemaker-rol", 
        "bucket_name": "igz-aws-kdl-training",
        "image_url": image_uri,
        "region_name": region_name,
        "instance_type": "ml.m5.4xlarge",
        "instance_count": 1,
        "synchronous": True,
        "timeout_seconds": 30000,
    }

    # Model Deployment
    client.create_deployment(
        name=app_name,
        model_uri=model_uri,
        flavor="python_function",
        config=config
    )
    print(f"Modelo desplegado en SageMaker con nombre {app_name}")
