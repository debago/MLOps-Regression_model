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



# Create mlartifacts folder inside VM and give permissions:

# This will store artifacts, models etc
mkdir -p /home/azureuser/mlartifacts
sudo chmod -R 777 /home/azureuser/mlartifacts

# This will store runs, expermients, metrics etc
mkdir -p /home/azureuser/mldb
Give permission-->
Check container user->

docker exec -it <mlflow-container> id
docker exec -it mlflow id
>> respinse uid=1000 gid=1000
sudo chown -R 1000:1000 /home/azureuser/mldb
sudo chmod -R 777 /home/azureuser/mldb



# Run mlflow container:

docker run -d -p 5000:5000 --name mlflow mlflow-server

# With Mount:
docker run -d -p 5000:5000 -v /home/azureuser/mlartifacts:/mlflow/martifacts -v /home/azureuser/mldb:/mlflow/db --name mlflow mlflow-server

docker run -d -p 5000:5000 \
    --name mlflow
    -v /home/azureuser/mlartifacts:/mlflow/artifacts \
    -v /home/azureuser/mldb:/mlflow/db \
    mlflow-server \
    server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow/db/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --allowed-hosts "*"

# oneline :

---------------------------------

docker run -d -p 5000:5000  --name mlflow     -v /home/azureuser/mlartifacts:/mlflow/artifacts     -v /home/azureuser/mldb:/mlflow/db     mlflow-server     server     --host 0.0.0.0     --port 5000     --backend-store-uri sqlite:///mlflow/db/mlflow.db     --default-artifact-root /mlflow/artifacts --allowed-hosts "*"
------------------------------------------

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
apt-get update && apt-get install net-tools curl -y
ss -tuln | grep 5000
curl http://localhost:5000
if curl not available:
python -c "import requests; print(requests.get('http://localhost:5000').status_code)"
expected response 200

python -c "import requests; print(requests.get('http://localhost:5000').text[:200])"

# Build Fastapi build image:

docker build -t ml-api ./api

# Start FastAPI Container:

docker run -d -p 8000:8000 \
    --name ml-api-container \
    -v /home/azureuser/mlartifacts:/mlflow/martifacts \
    -e MLFLOW_TRACKING_URI=http://<vm-ip>:5000 \
    -e MODEL_URI=models:/iris-model@latest \
    ml-api

docker run -d -p 8000:8000 \
    --name ml-api-container \
    -v /home/azureuser/mlartifacts:/mlflow/martifacts \
    -e MLFLOW_TRACKING_URI=http://40.75.103.57:5000 \
    -e MODEL_URI=models:/iris-model@latest \
    ml-api


# docker commands:

docker ps
docker ps -a
docker stop <container>
docker rm <container>
docker rm -f <container>
docker images
docker rmi <image-id>
docker stop <container>
docker restart <container>

# Debug as API can't load model:
docker exec -it <api-container> sh

echo $MLFLOW_TRACKING_URI 

>> Should return correct URI 

python -c "import requests; r = requests.get('http://<vm-ip>:5000'); print(r.status_code)"

>> response expected 200

run python

then inside run below to list registered models:

# list all models in MLflow 3.x
----------------------------------------------------
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Returns a list of registered models
models = client.search_registered_models()
for m in models:
    print(f"Model name: {m.name}")
    for v in m.latest_versions:
        print(f"  Version: {v.version}, Stage: {v.current_stage}")

-----------------------------------------------------------

# Loading the latest model:

-------------------------------

import mlflow.pyfunc

# Load the latest version
model_uri = "models:/iris-model/@latest"  # or just "models:/iris-model"
model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully!")

-------------------------------------
# Find model name: run inside api container

from mlflow.tracking import MlflowClient

client = MlflowClient()
models = client.search_registered_models()
for m in models:
    print(f"Model name: {m.name}")

------------------------------------------
# Check inside mlflow container:

from mlflow.tracking import MlflowClient
from mlflow.tracking import MlflowClient
client = MlflowClient()
models = client.search_registered_models()
print([m.name for m in models])

----------------------------------------------

docker exec -it <api-container> python
docker exec -it <api-container> sh