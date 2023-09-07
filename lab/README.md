
---

# Azure KDL Project Template

## Table of contents
- [About](#about)
- [Environment Setup](#environment-setup)
- [Execution](#execution)

## About 

This project serves as an integration between the **kdl-template** and **Azure services**, offering a streamlined process to manage machine learning workflows.

1. **Data Preparation**:
 Raw data undergoes processing to be transformed into a format suitable for modeling. Once processed, the data is securely stored in Azure Blob Storage, ensuring optimal availability and scalability.

2. **Model Training**:
 Once the data is ready, it serves as the foundation for training machine learning models. Utilizing Azure Machine Learning service, models are not only trained but also fine-tuned for peak performance in real-world scenarios.

3. **Prediction Retrieval**:
The culmination of the pipeline is in the retrieval of predictions. The trained models, residing in Azure Machine Learning, facilitate swift and accurate predictions.

The primary objective of this project is to simplify the machine learning lifecycle by harnessing the power and flexibility of Azure services, ensuring a smooth transition from data processing to model deployment and inference.

## Environment Setup

### Initial Azure Configuration

To set up and use the project correctly, specific values from Azure need to be obtained. Details on Azure specific configurations should be added here.

### Local Configuration

For local execution, environment variables should be defined in a **.env** file located inside the **processes/** directory. This file should contain appropriate configurations for Azure services.

Before executing scripts locally, ensure that all necessary configurations are defined. This includes specifying settings in `processes/configs/config_azure.ini`.

## Execution
To execute the data preparation process:

- Navigate to the project root directory.
- Run the command `make ingest-blob`. This command will:
    - Set up the environment.
    - Read the configuration from config_azure.ini.
    - Run the main.py script inside `prepare_data` to process and upload data to Azure Blob Storage.

To train standard classifiers:

- Navigate to the project root directory.
- Run the command `make train-classifiers`. This command will:
    - Set up the environment.
    - Read the configuration from config_azure.ini.
    - Run the main.py script inside `train_standard_classifiers` to train models using Azure Machine Learning.

To get predictions using standard classifiers:

Details on how to retrieve predictions using Azure services should be added here.

---
