# MLOps-Regression_model
End-to-End MLOps system (FastAPI +MLflow+Docker + Terrafom)

## Project Structure:

mlops-project/
│
├── src/
|   |── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── api/
│   ├── app.py          # FastAPI app
│   ├── model_loader.py  # Load MLflow model
|   ├── Dockerfile
│   └── requirements.txt
│
|── mlflow/              # MLflow service 👈 HERE
│   └── Dockerfile
|
|── data/
├── docker-compose.yml
│
|
├── requirements.txt
├── terraform/
│   ├── main.tf
│
├── params.yaml
├── README.md


#useful commands:
import src.preprocess import preprocess
python -m src.train 
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0 \
    --port 5000

# run fastapi:

1. uvicorn api.app:app --reload
2. uvicorn api.app:app --host 0.0.0.0 -port 8000 --reload
#---swagger UI
3. 127.0.0.1:8000/docs


# Build mlflow docker image:

docker build -t mlflow-server ./mlflow

# Build Fastapi build image:

docker build -t ml-api ./api

# Run mlflow container:

docker run -d -p 5000:5000 --name mlflow mlflow-server

# Run Mlfow API container:



# Start FastAPI Container:

docker run -d -p 8000:8000 \
    --name ml-api \
    -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    -e MODEL_URI=models:/iris-rf-model/Production \
    ml-api


