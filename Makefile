.PHONY: setup start-server start-worker deploy train predict

PROJECT_DIR := $(shell pwd)
DATA_PATH := $(PROJECT_DIR)/abalone.csv
ARTIFACTS_DIR := $(PROJECT_DIR)/src/web_service/local_objects
MODEL_PATH := $(ARTIFACTS_DIR)/model.pkl
PREPROC_PATH := $(ARTIFACTS_DIR)/preprocessor.pkl

setup:
	source .venv/bin/activate && pip install -e .

start-server:
	prefect server start

start-worker:
	prefect work-pool create --type process mlops-pool || true
	prefect worker start --pool mlops-pool

deploy:
	python -m prefect deploy src/modelling/workflow.py:train_model_workflow --name train-model --pool mlops-pool
	python -m prefect deploy src/modelling/workflow.py:predict_flow --name batch-predict --pool mlops-pool

train:
	python -m prefect deployment run "Train model/train-model" \
	  -p data_path="$(DATA_PATH)" \
	  -p artifacts_filepath="$(ARTIFACTS_DIR)"

predict:
	python -m prefect deployment run "Batch predict/batch-predict" \
	  -p input_filepath="$(DATA_PATH)" \
	  -p model_path="$(MODEL_PATH)" \
	  -p preproc_path="$(PREPROC_PATH)"
