# AWS KDL Project Template - Prepare Data 

## Table of contents
- About prepare_data
- Environment Setup
- Configuration
- Execution
- Testing

## About 
The prepare_data component is responsible for preparing and processing data for further analysis and machine learning tasks. This includes tasks like cleaning data, transforming data formats, and uploading processed data to AWS S3.

## Environment Setup
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

**Local Configuration**

For local execution, environment variables should be defined in a **.env** file located inside the **lab/processes/** directory. This file should contain the following variables:

```# IAM
AWS_ACCESS_KEY_ID=<your_access_key_id>
AWS_SECRET_ACCESS_KEY=<your_secret_access_key>
AWS_REGION_NAME=<your_region_name>

# S3
AWS_S3_BUCKET_NAME=<your_s3_bucket_name>

# SAGEMAKER
AWS_SAGEMAKER_ROLE_NAME=<your_role_name>
AWS_SAGEMAKER_IMAGE_URI=<your_image_uri>
AWS_SAGEMAKER_ROLE_ARN=<your_role_arn>
AWS_REGION_NAME=<your_region_name> 
```

## Execution
To execute the data preparation process:

- Navigate to the lab directory.
- Run the command **make ingest-s3**. This command will:
    - Set up the environment.
    - Read the configuration from config_aws.ini.
    - Run the main.py script inside prepare_data to process and upload data to AWS S3.

## Testing
#### (TO DO)
Testing ensures the reliability of the data preparation process. To run tests:

1. Navigate to the prepare_data directory.
2. Run the command <specific-command-for-tests> (Replace <specific-command-for-tests> with the appropriate command if testing is implemented).
