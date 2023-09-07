# AWS KDL Project Template

## Table of contents
- [About](#about)
- [Environment Setup](#environment-setup)
- [Execution](#execution)


## About 
About
The project serves as an integration between the **kdl-template** and **AWS services**, offering a streamlined process to manage machine learning workflows.

1. **Data Preparation**:
 Raw data undergoes processing to be transformed into a format suitable for modeling. Once processed, the data is securely stored in AWS S3, ensuring optimal availability and scalability.

2. **Model Training**:
 Once the data is ready, it serves as the foundation for training machine learning models. Utilizing AWS SageMaker, models are not only trained but also fine-tuned for peak performance in real-world scenarios.

3. **Prediction Retrieval**:
The culmination of the pipeline is in the retrieval of predictions. The trained models, residing in AWS SageMaker, facilitate swift and accurate predictions.

This project's primary objective is to simplify the machine learning lifecycle by harnessing the power and flexibility of AWS services, ensuring a smooth transition from data processing to model deployment and inference.

## Environment Setup

### Initial AWS Configuration

To set up and use the project correctly, specific values from AWS need to be obtained. Below is a guide on how to fetch them:
1. **AWS SageMaker Image URI**

    The Image URI is an identifier for the Docker container you wish to use with SageMaker. If you're using a custom container, you'll need to build and push it to AWS Elastic Container Registry (ECR). Here's how to obtain the URI:

    - **Setting Up AWS CLI**

        Before you can push an image to ECR or work with SageMaker, you need to configure AWS CLI with the right credentials:
    
        1. Ensure you have the AWS CLI installed.
        2. Run aws configure and provide the access key, secret key, region, and desired output format for a user with appropriate ECR and SageMaker permissions.

    - **Building and Pushing the Docker Image**

        1. Run the command:

            ```
            mlflow sagemaker build-and-push-container
            ``` 
        
            This command will build the Docker image for you and push it to ECR.
        
        2. Upon successful completion, the command will provide a URI for the newly pushed Docker image.

            Set this URI as your `AWS_SAGEMAKER_IMAGE_URI`.

2. **AWS SageMaker Role ARN**

    The Role ARN is the identifier for the AWS Identity and Access Management (IAM) role that grants SageMaker the required permissions to access other AWS services. To fetch the Role ARN:

    1. Navigate to the AWS IAM Dashboard.
    2. In the left navigation pane, select Roles.
    3. Search for and select the role you wish to use with SageMaker.
    4. In the role summary page, copy the Role ARN.
    Set this ARN as your `AWS_SAGEMAKER_ROLE_ARN`.

3. **IAM User**

    For the successful execution of the project, specific AWS services are leveraged. To ensure seamless integration and operation, it's crucial to set up the necessary permissions for the AWS user. Here's a step-by-step guide to achieving this:

    - **Amazon S3 Permissions**:
        Navigate to the AWS Management Console.
        Open the IAM dashboard and select "Users".
        Choose the desired user and navigate to the "Permissions" tab.
        Attach a custom policy that allows:
        - To upload files to a bucket.
        - To read files from a bucket

    - **Amazon ECR Permissions**:
        In the user's "Permissions" tab, attach a custom policy that allows:
        - To upload Docker images.
        - To retrieve Docker images for deployment.

    - **Amazon SageMaker Permissions**:
        Still in the "Permissions" tab, attach a custom policy that includes permissions for:
        - To upload trained models.
        - To set up an inference endpoint.
        - To get predictions from the model..



### Local Configuration

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

Before executing scripts locally, ensure that all necessary configurations are defined. This includes specifying settings in `lab/processes/configs/config_aws.ini` and setting environment variables:



    - [paths]: Defines paths like the processed data directory.
    - [mlflow]: Contains settings related to MLflow.
    - [filenames]: Defines filenames for specific artifacts like models or plots.
    - [training]: Contains settings for training processes, like batch size and learning rate.

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





