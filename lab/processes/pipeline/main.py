from azureml.pipeline.core import Pipeline
from azureml.core import Experiment
from commands import load_data_from_blob, train_standard_classifier
from lab.processes.azure.workspace import AzureWorkspaceConnector

def main():
    azure_connector = AzureWorkspaceConnector()
    ws = azure_connector.ws
    compute_target = 'cpu-cluster'

    data_load_step, X_train_data, X_val_data, X_test_data, y_train_data, y_val_data, y_test_data = load_data_from_blob(compute_target, ws)
    
    train_step = train_standard_classifier(compute_target, X_train_data, X_val_data, X_test_data, y_train_data, y_val_data, y_test_data)

    pipeline = Pipeline(workspace=ws, steps=[data_load_step, train_step])
    
    print("Pipeline is built with name: breast_cancer_pipeline")
    pipeline.validate()
    print("Pipeline validation complete")
    pipeline_run = Experiment(ws, 'my_pipeline_run').submit(pipeline)
    print("Pipeline is published.")

if __name__ == "__main__":
    main()










