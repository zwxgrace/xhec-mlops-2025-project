## 🛠️ Project Setup, Quality & Continuous Integration (CI)
This project adopts modern MLOps practices by integrating robust dependency management, automated code quality checks, and a Continuous Integration pipeline.

① pyproject.toml (Dependency Management)

Purpose: Defines all project dependencies and configuration settings.

Key Effect: pydantic is added as a core runtime dependency；pre-commit and ruff are configured as development dependencies (within the dev group) for local tooling.

② pre-commit-config.yaml (Local Code Quality Checks)

Purpose: Configures the Git hook manager (pre-commit) to run checks automatically before every commit.

Key Effect: Defines a series of hooks (e.g., trailing-whitespace, check-yaml, ruff, ruff-format).These checks enforce code style and catch simple errors early, preventing bad code from entering the repository history.

③ github/workflows/ci.yaml (Continuous Integration Workflow)

Purpose: Sets up the CI/CD pipeline using GitHub Actions.

Key Effect: Triggers an automated workflow on every push and pull_request. The workflow installs dependencies, runs all pre-commit checks, and executes pytest to ensure code functionality and adherence to quality standards before merging.

### Train
```bash
uv run python -m src.modelling.main ./abalone.csv
```

### 🐚 Abalone Prediction API — Deployment & Workflow Guide

This guide walks you through the complete process of building the Docker image, running the container, deploying Prefect workflows, training the model, generating predictions, and checking your results — all in a clear, step-by-step manner.


### ⚙️ Environment Overview

| Component            | Description                      | Port (Container → Host)               |
| -------------------- | -------------------------------- | ------------------------------------- |
| **FastAPI**          | Online prediction API            | 8001 → **8000**                       |
| **Prefect Orion UI** | Workflow orchestration dashboard | 4201 → **4200**                       |
| **Model Artifacts**  | Trained model & preprocessor     | `/app/src/web_service/local_objects/` |


### 🚀 Step 1. Build Docker Image

We start by building the image and launching your container. This will create an isolated environment where both FastAPI and Prefect can run together.

```bash
docker build --no-cache -t abalone-prediction-api -f Dockerfile.app .
```
✅ Expected output:
```
Successfully built <IMAGE_ID>
Successfully tagged abalone-prediction-api:latest
```

### 🧩 Step 2. Run the Container

Then, you'll run the container in docker.

```bash
docker run -d -p 8000:8001 -p 4200:4201 --name abalone_api_service abalone-prediction-api
```
✅ Expected output:
```
<CONTAINER_ID>
```
At this point, you’re inside the container environment (root@/app) — all subsequent deployment commands will be run here.

## Check container is running:
```bash
docker ps
```
You should see something like:
CONTAINER ID   IMAGE                     PORTS                                        NAMES
abcd1234...    abalone-prediction-api    0.0.0.0:8000->8001/tcp, 0.0.0.0:4200->4201/tcp   abalone_api_service


### 🧠 Step 3. Enter the Container & Deploy Prefect Workflows

Once the container is up, enter it to interact with Prefect directly:

```bash
docker exec -it abalone_api_service bash
```

Inside the container you should see:
```
root@<container_id>:/app#
```

All commands below are executed inside the container (root@/app).
Next, you’ll deploy two Prefect workflows: one for training the model and one for batch prediction.
These flows will appear in the Prefect UI after deployment.

## (A) Deploy the Training Workflow

```bash
export PREFECT_API_URL='http://localhost:4201/api'
prefect deploy src/modelling/workflow.py:train_model_workflow -n "Model Retraining Deployment" --pool "default"
```

You should now see this in Prefect UI →
👉 http://localhost:4200
 → Deployments → Train model / Model Retraining Deployment

 ## (B) Deploy the Batch Prediction Workflow

 ```bash
 export PREFECT_API_URL='http://localhost:4201/api'
prefect deploy src/modelling/workflow.py:predict_flow -n "Batch Prediction" --pool "default"
```

Now you’ll see both deployments in Prefect UI:

Train model / Model Retraining Deployment
Batch predict / Batch Prediction


### 🧮 Step 4. Run the Training Deployment

Now let’s actually train the model. This flow will read the abalone.csv dataset, train a regression model, and save the resulting files inside /app/src/web_service/local_objects/.

```bash
export PREFECT_API_URL='http://localhost:4201/api'
prefect deployment run 'Train model/Model Retraining Deployment' \
  -p data_path='/app/abalone.csv' \
  -p artifacts_filepath='/app/src/web_service/local_objects'
```

When prompted:
| Question               | Answer | Meaning                          |
| ---------------------- | ------ | -------------------------------- |
| Pull code from remote? | **n**  | Code already in the container    |
| Configure schedule?    | **n**  | We'll schedule later manually    |
| Save configuration?    | **y**  | So future deployments are faster |

During the run, Prefect will log the progress and metrics (like RMSE, MAE, R²).



### 🧭 Step 5. Run the Batch Prediction Flow

After training, you can generate predictions using the new model.
This step can be executed from your host machine (no need to re-enter the container).

(If you exit the container, you need to enter it again.)
```bash
docker exec -it abalone_api_service bash
```
Then,

```bash
cd /app
export PREFECT_API_URL='http://localhost:4201/api'
prefect deployment run 'Batch predict/Batch Prediction' \
  -p input_filepath='/app/abalone.csv' \
  -p model_path='/app/src/web_service/local_objects/model.pkl' \
  -p preproc_path='/app/src/web_service/local_objects/preprocessor.pkl' \
  -p output_path='/app/src/web_service/local_objects/batch_predictions.csv'
```

✅ Expected output:
```
Flow run 'Batch predict/Batch Prediction' submitted for execution.
Running task 'predict_batch'...
Predictions saved to /app/src/web_service/local_objects/batch_predictions.csv
Flow run completed successfully!
```

You’ve now generated predictions using the trained model.


### 🔁 Step 6. (Optional) Schedule Automatic Retraining

If you want your model to automatically retrain every hour, you can add a simple schedule:
```bash
docker exec -w /app abalone_api_service bash -c \
"export PREFECT_API_URL='http://localhost:4201/api' && \
prefect deploy src/modelling/workflow.py:train_model_workflow \
  -n 'Model Retraining Deployment' --pool 'default' --interval '3600'"
```

Prefect will confirm:
Deployment 'Train model/Model Retraining Deployment' updated with interval schedule (every 3600 seconds)

In the Prefect UI, a 🕒 icon next to the deployment name indicates that it’s scheduled automatically.


### Step 7. Check Results and Outputs

Once the workflows are complete, you can view their results in several ways:

| What to check                 | Where to look                                                | Expected Output                 |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------- |
| **Training status / metrics** | Prefect UI → *Train model → Flow Runs*                       | RMSE, MAE, R² metrics in logs   |
| **Prediction output file**    | `/app/src/web_service/local_objects/batch_predictions.csv`   | CSV with predicted ages         |
| **FastAPI health**            | [http://localhost:8000/health](http://localhost:8000/health) | `{"ready": true}`               |
| **FastAPI docs**              | [http://localhost:8000/docs](http://localhost:8000/docs)     | Interactive `/predict` endpoint |


### Steo 8. Test the Prediction API

Once your model is trained and the FastAPI service is running, you can test the prediction endpoint directly using curl.

Make sure you use the correct field names (Sex, Length, Diameter, etc.) — they must match exactly what the model expects in your preprocessing pipeline.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Sex":"M","Length":0.45,"Diameter":0.36,"Height":0.11,"Whole_weight":0.95,"Shucked_weight":0.42,"Viscera_weight":0.19,"Shell_weight":0.30}'
```

🔹 Expected Response

If the API and model are working correctly, you should receive a JSON object like this:

{
  "predictions": [10.72],
  "n_samples": 1
}

This confirms:

The FastAPI service is running and reachable at port 8000

Your trained model and preprocessor were successfully loaded

The input data was parsed and processed correctly

You can also open the interactive API documentation at http://localhost:8000/docs, click POST /predict, paste the same JSON body, and test it through the Swagger UI.


### Step 9. Retrieve or Inspect Saved Artifacts

To verify your generated files inside the container:

```bash
docker exec -it abalone_api_service ls /app/src/web_service/local_objects
```

You should see:

model.pkl
preprocessor.pkl
batch_predictions.csv

If you wish to copy the prediction file to your host:
```bash
docker cp abalone_api_service:/app/src/web_service/local_objects/batch_predictions.csv ./batch_predictions.csv
```



### 🔚 Final Notes

At this stage:

Your Docker container hosts both Prefect and FastAPI services.

Prefect orchestrates model training and batch prediction.

FastAPI provides online prediction through /predict.

You can monitor training progress, view logs, and inspect metrics directly in the Prefect UI at http://localhost:4200.

When finished, you can clean up the environment:

```bash
docker stop abalone_api_service && docker rm abalone_api_service
```


### ✅ Congratulations!
You have successfully built, deployed, trained, and served a machine learning model using Docker, Prefect, and FastAPI — all in one reproducible workflow.
