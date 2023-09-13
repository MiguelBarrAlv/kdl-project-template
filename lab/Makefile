BASE_PATH = $(shell pwd)
PATH_CONFIG = $(BASE_PATH)/processes/configs/config_aws.ini
PREPARE_DATA_PATH = $(BASE_PATH)/processes/prepare_data
TRAIN_CLASSIFIERS_PATH = $(BASE_PATH)/processes/train_standard_classifiers
MLRUNS_PATH = $(BASE_PATH)/mlruns
USAGE_EXAMPLES_PATH = $(BASE_PATH)/usage_examples

.PHONY: ingest-s3 train-classifiers get-prediction

ingest-s3:
	@echo "Uploading training data to S3..."
	@cd $(PREPARE_DATA_PATH) && pipenv sync
	@cd $(PREPARE_DATA_PATH) && PATH_CONFIG=$(PATH_CONFIG) pipenv run python3 main.py

train-classifiers:
	@echo "Training standard classifiers..."
	@cd $(TRAIN_CLASSIFIERS_PATH) && pipenv sync
	@cd $(TRAIN_CLASSIFIERS_PATH) && PATH_CONFIG=$(PATH_CONFIG) pipenv run python3 main.py
	@cd $(TRAIN_CLASSIFIERS_PATH) && pipenv run python3 main.py

get-prediction:
	@echo "Getting predictions using standard classifiers..."
	@cd $(USAGE_EXAMPLES_PATH) && pipenv sync
	@cd $(USAGE_EXAMPLES_PATH) && PATH_CONFIG=$(PATH_CONFIG) pipenv run python3 get_prediction_standard_classifiers.py
