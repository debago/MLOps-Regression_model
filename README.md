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

# Create mlartifacts folder inside VM and give permissions:

mkdir -p /home/azureuser/mlartifacts
sudo chmod -R 777 /home/azureuser/mlartifacts

# Run mlflow container:

docker run -d -p 5000:5000 --name mlflow mlflow-server

# With Mount:
docker run -d -p 5000:5000 -v /home/azureuser/mlartifacts:/mlflow/martifacts --name mlflow mlflow-server

# Debug:
docker run -p 5000:5000 mlflow-server \
server \ 
--host 0.0.0.0 \
--port 5000 \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root /mlflow/artifacts \
--allowed-hosts "*"

docker run -it --entrypoint sh <image-name>


# single line command:

docker run -p 5000:5000 mlflow-server server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts --allowed-hosts "*"


# Validate:
 Run inside vm:

curl http://localhost:5000

from local machine:

curl http:// vm-ip:5000

Go Inside container:

docker exec -it mlflow sh
ps aux | grep mlflow
netstat -tuln | grep 5000
if net-stat not available
pip install net-tools curl -y
ss -tuln | grep 5000
curl http://localhost:5000
if curl not available:
python -c "import requests; print(requests.get('http://localhost:5000').status_code)"
expected response 200

python -c "import requests; print(requests.get('http://localhost:5000').text[:200])"
# Start FastAPI Container:

docker run -d -p 8000:8000 \
    --name ml-api \
    -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    -e MODEL_URI=models:/iris-rf-model/Production \
    ml-api

# docker commands:

docker ps
docker ps -a
docker stop <container>
docker rm <container>
docker images
docker rmi <image-id>

# 
