# KDL Project Template

## Project structure

The project repository has the following directory structure:

```
├── lab
│   │
│   ├── analysis  <- Analyses of data, models etc. (typically notebooks)
│   │
│   ├── docs      <- High-level reports, executive summaries at each milestone (typically .md)
│   │
│   ├── lib       <- Importable functions shared between analysis notebooks and processes scripts
│   │                (including unit tests)
│   │
│   └── processes           <- Source code for reproducible workflow steps. For example:
│       ├── prepare_data   
│       │   ├── main.py      
│       │   ├── image_data.py  
|       │   └── test_image_data.py
|       ├── train_model
│       │   ├── main.py      
│       │   ├── convnet.py  
|       │   └── test_convnet.py
│       └── ...
│   
├── goals         <- Acceptance criteria (TBD)
│   
├── runtimes      <- Code for generating deployment runtimes (.krt)
│   
├── .drone.yml    <- Instructions for Drone runners
├── .flake8     
├── .gitignore
|
└── README.md
```


## Example project pipelines

KDL contains various components that need to be correctly orchestrated and connected. 
To illustrate their intended usage, we provide two example machine learning pipelines already implemented in KDL. 
The first example pipeline is a simple classification problem with standard ML models from scikit-learn.
The second example pipeline is an image classification problem addressed with convolutional networks implemented in PyTorch.

### Scikit-learn example: Wine classification

The first example pipeline is a simple classification task. 
Based on the [Wine Recognition Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset), the aim is to classify three different types of wines based on their physicochemical characteristics (alcohol content, malic acid, etc.).

The code for wine classification is in [lab/processes/sklearn_example/main.py](lab/processes/sklearn_example/main.py).

The execution of the wine classification pipeline on Drone agents is specified in [.drone.yml](.drone.yml) (for simplicity, we are omitting various additional components, such as the environment variables and the AWS secrets):

```yaml
kind: pipeline
type: kubernetes
name: application-examples

trigger:
  ref:
  - refs/tags/run-examples-*

steps:
  - name: sklearn-example
    image: terminus7/sci-toolkit-runner:1.1.2
    commands:
      - python3 lab/processes/sklearn_example/main.py
```

To trigger the execution of this pipeline on Drone runners, push a tag containing the name matching the trigger (e.g. in this case, `run-examples-v1`) to the remote repository.
For more information, see the section Launching experiment runs (Drone) below.

The results of executions are stored in MLflow: 
in the simplified example of wine classification, we are only tracking one parameter (name of the classifier), and one metric (the obtained validation accuracy). 
In a real-world project, you are likely to be tracking many parameters and metrics of interest.
The connection to MLflow to log these parameters and metrics is established via the code in the [main.py](lab/processes/sklearn_example/main.py) and with the environment variables in [.drone.yml](.drone.yml). For more information, see the section "Logging experiment results (MLflow)" below.
To see the tracked experiments, visit the MLflow tool UI.


### PyTorch example: digit classification

The second example pipeline is based on an image classification problem, with the aim of classifying digits from the [Optical Recognition of Handwritten Digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset) dataset. This is _not_ the standard MNIST dataset (over 20,000 images of 28x28 pixels), it is a considerably smaller dataset (1797 images of 8x8 pixels), with the advantage that it does not require downloading from the internet as it is already distributed with the scikit-learn library in the `sklearn.datasets` package.

This example pipeline, defined by the code in [.drone.yml](.drone.yml) (pipeline `pytorch-example`) and in [lab/processes/pytorch_example](lab/processes/pytorch_example), contains of the following steps:
- **prepare_data:** the dataset images are loaded (from sklearn), normalized and transformed; the transformed data are split into train, validation and test sets; and the processed data are stored on the shared volume.
- **train_model:** a simple convolutional neural network is trained to classify digits 0-9 on the training data. The training history (accuracy and loss per epoch on both training and validation data) are stored as an artifact in MLflow (`training_history.csv` and visualized in `.png`). The model with the highest validation accuracy is saved as a .joblib file in MLflow artifacts, and is used to produce an assessment of model performance on the validation dataset (e.g. saving the loss and accuracy metrics, and the confusion matrix of the validation set, `confusion_matrix.png`, all logged to MLflow).
- **test_model:** finally, the trained model is validated against the withheld test dataset. For simplicity of the example, this step is placed in the same pipeline with data preparation and model training. However, in reality this would only be carried once in the project to avoid "test set leakage", so you will probably want to separate this step from the previous steps into a stand-alone Drone pipeline and only execute it once at the end of the development phase of the project.

## Importing library functions

Reusable functions can be imported from the library (`lib` subdirectory) to avoid code duplication and to permit a more organized structuring of the repository.

To import library code in notebooks, you may need to add the `lab` directory to PYTHONPATH, for example as follows:

```python
import sys
from pathlib import Path

DIR_REPO = Path.cwd().parent.parent
DIR_LAB = DIR_REPO / "lab"

sys.path.append(str(DIR_LAB))

from lib.viz import plot_confusion_matrix
```

To be able to run imports from the `lib` directory on Drone, you may add it to PYTHONPATH in .drone.yml as indicated:

```yaml
environment:
  PYTHONPATH: /drone/src/lab
```

`/drone/src` is the location on the Drone runner that the repository is cloned to, and `lab` is the name of the laboratory section of our repository which includes `lib`. 
This then allows importing library functions directly from the Python script that is being executed on the runner, for instance:

```python
from lib.viz import plot_confusion_matrix
```

To see a working example, refer to the existing `application-examples` pipeline defined in .drone.yml 
(the PyTorch example pipeline uses library imports in `processes/pytorch_example/main.py`).


## Launching experiment runs (Drone)

To enable full tracability and reproducibility, all executions that generate results or artifacts 
(e.g. processed datasets, trained models, validation metrics, plots of model validation, etc.) 
are run on Drone runners instead of the user's Jupyter or Vscode tools. 

This way, any past execution can always be traced to the exact version of the code that was run (`VIEW SOURCE </>` in the UI of the Drone run)
and the runs can be reproduced with a click of the button in the UI of the Drone run (`RESTART`).

The event that launches a pipeline execution is defined by the trigger specified in .drone.yml. 
An example is shown below:

```yaml
trigger:
  ref:
  - refs/tags/process-data-*
```

With this trigger in place, the pipeline will be executed on Drone agents whenever a tag matching the pattern specified in the trigger is pushed to the remote repository, for example:

```bash
git tag process-data-v0
git push origin process-data-v0 
```

Note: If using an external repository (e.g. hosted on Github), a delay in synchronization between Gitea and the mirrored external repo may cause a delay in launching the pipeline on the Drone runners. 
This delay can be overcome by manually forcing a synchronization of the repository in the Gitea UI Settings.

## Logging experiment results (MLflow)

To compare various experiments, and to inspect the effect of the model hyperparameters on the results obtained, you can use MLflow experiment tracking. Experiment tracking with MLflow enables logging the parameters with which every run was executed and the metrics of interest, as well as any artifacts produced by the run. 

The environment variables for connecting to MLflow server are provided in .drone.yml:

```yaml
environment:
  MLFLOW_URL: http://mlflow-server:5000
  MLFLOW_S3_ENDPOINT_URL: http://{{ ProjectID }}:9000
  MLFLOW_EXPERIMENT: {{ ProjectID }}
```

The usage of MLflow for experiment tracking is illustrated by the scikit-learn example pipeline in [lab/processes/sklearn_example/main.py](lab/processes/sklearn_example/main.py).

```python
import mlflow

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name=MLFLOW_RUN_NAME):

    # (... experiment code ...)

    # Log to MLflow
    mlflow.log_param("classifier", model_name)
    mlflow.log_metric("validation_accuracy", val_acc)
```

For more information on logging data to runs, see [MLflow documentation on logging](https://www.mlflow.org/docs/latest/tracking.html#logging-data-to-runs).

Whenever one script execution trains various models (e.g. in hyperparameter search, where a model is trained with many different combinations of hyperparameters), it is helpful to use nested runs. This way, the sub-runs will appear grouped under the parent run in the MLflow UI:

```python
import mlflow

with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
    
    # (... experiment setup code shared between subruns ... )

    with mlflow.start_run(run_name=SUBRUN_NAME, nested=True):
        
        # (... model training code ...)
        
        mlflow.log_metric("classifier", model_name)
        mlflow.log_param("validation_accuracy", val_acc)
```

To compare the executions and vizualise the effect of logged parameters on the logged metrics, 
you can select the runs you wish to compare in the MLflow UI, select "Compare" and add the desired parameters and metrics to the visualizations provided through the UI. 
Alternatively, the results can also be queried with the MLflow API. For more information on the latter, see [MLflow documentation on querying runs](https://www.mlflow.org/docs/latest/tracking.html#querying-runs-programmatically).


