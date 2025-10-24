#!/bin/bash

echo "--- Starting Prefect server (in background) ---"
prefect server start --host 0.0.0.0 --port 4201 &

export PREFECT_API_URL="http://localhost:4201/api"

echo "--- Waiting 10 seconds for Prefect Server to initialize... ---"
sleep 10

echo "--- Starting Prefect worker (in background) ---"
prefect worker start --pool "default" --work-queue "default" &

echo "--- Waiting 10 seconds for services to initialize... ---"
sleep 10

echo "--- Starting FastAPI (Uvicorn) server (in foreground) ---"
uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
