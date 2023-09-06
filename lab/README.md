# AWS KDL Project Template - Prepare Data 

## Table of contents
- About prepare_data
- Environment Setup
- Configuration
- Execution
- Testing

## About 
The prepare_data component is responsible for preparing and processing data for further analysis and machine learning tasks. This includes tasks like cleaning data, transforming data formats, and uploading processed data to AWS S3.

## Local Environment Setup
Before running any script inside prepare_data, make sure to set up the environment:

1. Navigate to the prepare_data directory.
2. Run pipenv sync to ensure all dependencies are installed.
3. Activate the virtual environment with pipenv shell.
## Configuration
Before executing scripts locally, ensure that all necessary configurations are defined. This includes specifying settings in config_aws.ini and setting environment variables.

**Config_aws.ini**

    - [paths]: Defines paths like the processed data directory.
    - [mlflow]: Contains settings related to MLflow.
    - [filenames]: Defines filenames for specific artifacts like models or plots.
    - [training]: Contains settings for training processes, like batch size and learning rate.

**Initial AWS Configuration**
To set up and use the project correctly, specific values from AWS need to be obtained. Below is a guide on how to fetch them:
1. **AWS SageMaker Image URI**

    The Image URI is an identifier for the Docker container you wish to use with SageMaker. If you're using a custom container, you'll need to build and push it to AWS Elastic Container Registry (ECR). Here's how to obtain the URI:

    - **Setting Up AWS CLI**

        Before you can push an image to ECR or work with SageMaker, you need to configure AWS CLI with the right credentials:
    
        1. Ensure you have the AWS CLI installed.
        2. Run aws configure and provide the access key, secret key, region, and desired output format for a user with appropriate ECR and SageMaker permissions.

    - **Building and Pushing the Docker Image**

        1. Run the command mlflow sagemaker build-and-push-container. This command will build the Docker image for you and push it to ECR.
        2. Upon successful completion, the command will provide a URI for the newly pushed Docker image.

        Set this URI as your `AWS_SAGEMAKER_IMAGE_URI`.

2. **AWS SageMaker Role ARN**

    The Role ARN is the identifier for the AWS Identity and Access Management (IAM) role that grants SageMaker the required permissions to access other AWS services. To fetch the Role ARN:

    1. Navigate to the AWS IAM Dashboard.
    2. In the left navigation pane, select Roles.
    3. Search for and select the role you wish to use with SageMaker.
    4. In the role summary page, copy the Role ARN.
    Set this ARN as your `AWS_SAGEMAKER_ROLE_ARN`.


**Local Configuration**

For local execution, environment variables should be defined in a **.env** file located inside the **lab/processes/** directory. This file should contain the following variables:

```
# IAM
AWS_ACCESS_KEY_ID=<your_access_key_id>
AWS_SECRET_ACCESS_KEY=<your_secret_access_key>

# S3
AWS_S3_BUCKET_NAME=<your_s3_bucket_name>

# SAGEMAKER
AWS_SAGEMAKER_IMAGE_URI=<your_image_uri>
AWS_SAGEMAKER_ROLE_ARN=<your_role_arn>
AWS_REGION_NAME=<your_region_name> 
```

## Execution
To execute the data preparation process:

- Navigate to the lab directory.
- Run the command `make ingest-s3`. This command will:
    - Set up the environment.
    - Read the configuration from config_aws.ini.
    - Run the main.py script inside prepare_data to process and upload data to AWS S3.

This command prepares the data, reads configurations from config_aws.ini, and then uploads the processed data to the specified AWS S3 bucket.

To train standard classifiers:

- Navigate to the lab directory.
- Run the command make `make train-classifiers`. This command will:
    - Set up the environment.
    - Read the configuration from config_aws.ini.
    - Run the main.py script inside train_standard_classifiers to train and deploy model in AWS Sagemaker.

To get predictions using standard classifiers:

- Navigate to the root directory of the project.
- Run the command `make get-prediction`. This command will:
    - Navigate to the usage_examples directory.
    - Set up the environment using the Pipfile located within.
    - Read the configuration from config_aws.ini.
    - Run the get_prediction_standard_classifiers.py script to obtain predictions using the trained model.

    This command retrieves predictions based on the model previously trained and deployed on AWS Sagemaker, leveraging the configurations set in config_aws.ini.





