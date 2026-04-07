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

-------------------------------------------------
# Observation:

1. Need to check whether mounts are correct.
2. Change the owner to the actual user from root before creation mount files and give properpermission.
3. Check Backend URI path be mindful whether it's relative of absolute path use 3 /// for relative
  and 4 //// for absolute path
4. If artifacts not visible  to the mount path check docker run command and use artifact_destination and serve-artifact

------------------------------------------


#useful commands:

# Login to azure vm

chmod 400 <private_key>
#chmod 400 ~/Downloads/vm.pem
ssh -i <private-pem key> azureuser@vm-public-ip
# ssh -i ~/Downloads/vm.pem azureuser@40.75.103.57

export MLFLOW_TRACKING_URI=http://40.75.103.57:5000
echo $MLFLOW_TRACKING_URI
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

export MLFLOW_TRACKING_URI=http://40.75.103.57:5000
echo $MLFLOW_TRACKING_URI

# Run mlflow container:

docker run -d -p 5000:5000 --name mlflow mlflow-server

# With Mount:
docker run -d -p 5000:5000 -v /home/azureuser/mlartifacts:/mlflow/martifacts -v /home/azureuser/mldb:/mlflow/db --name mlflow mlflow-server

docker run -d -p 5000:5000 \
  -u $(id -u):$(id -g) \
  -v /home/azureuser/mldb:/mlflow/db \
  -v /home/azureuser/mlartifacts:/mlflow/artifacts \
  --name mlflow \
  mlflow-server:latest \
  server \
    --backend-store-uri sqlite:////mlflow/db/mlflow.db \
    --artifacts-destination /mlflow/artifacts \
    --serve-artifacts \
    --allowed-hosts "*" \
    --host 0.0.0.0 \
    --port 5000




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
    -u $(id -u):$(id -g) \
    --name ml-api-container \
    -e MLFLOW_TRACKING_URI=http://<vm-ip>:5000 \
    -e MODEL_URI=models:/iris-model@latest \
    ml-api

docker run -d -p 8000:8000 \
    --name ml-api-container \
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

# remove all container which are stopped
docker rm $(docker ps -aq -f status=exited)

# run regularly for disk space management:

docker container prune -f
docker system prune -a   # Removed stopped container, unused images, build cache


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

# debug as mlflow is not visible with runs/models/artifacts

1. Step 1 — Confirm MLflow server config
docker ps
docker inspect mlflow-container | grep -A 20 Cmd
# inspect mount , volume none i above output is expected

docker inspect mlflow --format '{{json .Mounts}}'

2. Step 2 — Check actual DB usage

docker exec -it mlflow-container bash
ls /mlflow/db
ls /mlflow/artifacts

docker stop mlflow
docker rm mlflow

# Clean old local  runs
rm -rf mlruns

# ensure volume mounts:

mkdir -p /home/azureuser/mldb
mkdir -p /home/azureuser/mlartifacts
chmod -R 777 /home/azureuser/mldb
chmod -R 777 /home/azureuser/mlartifacts

ls -ld /home/azureuser/mlartifacts
ls -ld /home/azureuser/mldb

# if file doesn't have proper permission: 

change owner:

sudo chown -R azureuser:azureuser /home/azureuser/mlartifacts
sudo chown -R azureuser:azureuser /home/azureuser/mldb

Then give permission:

chmod -R 755 /home/azureuser/mlartifacts
mkdir -p /home/azureuser/data
sudo chown -R azureuser:azureuser /home/azureuser/data
chmod -R 755 /home/azureuser/data

if above fails:
---------------
sudo rm -rf /home/azureuser/mlartifacts
sudo rm -rf /home/azureuser/mldb

mkdir -p /home/azureuser/mlartifacts
mkdir -p /home/azureuser/mldb
-----------------------


# run docker :

docker run -d -p 5000:5000 \
  -v /home/azureuser/mldb:/mlflow/db \
  -v /home/azureuser/mlartifacts:/mlflow/artifacts \
  --name mlflow \
  mlflow-server:latest \
 server \
    --backend-store-uri sqlite:///mlflow/db/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --allow-hosts="*" \
    --host 0.0.0.0 \
    --port 5000

--------------------------------
# docker run with user ensures:

-u $(id -u):$(id -g)
Runs container as your VM user (azureuser)
Prevents root-owned files
Ensures:
/home/azureuser/mldb ✅ writable
/home/azureuser/mlartifacts ✅ writable
--------------------------------------
docker run -d -p 5000:5000 \
  -u $(id -u):$(id -g) \
  -v /home/azureuser/mldb:/mlflow/db \
  -v /home/azureuser/mlartifacts:/mlflow/artifacts \
  --name mlflow \
  mlflow-server:latest \
  server \
    --backend-store-uri sqlite:////mlflow/db/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --allowed-hosts "*" \
    --host 0.0.0.0 \
    --port 5000
------------------------

# How to confirm Entrypoint:

docker inspect mlflow-server:latest | grep -A 5 Entrypoint
docker inspect mlflow-container | grep -A 20 Cmd

# if container exit immediately after startup:
docker logs mlflow
docker inspect mlflow --format '{{.State.Status}} {{.State.ExitCode}} {{.State.Error}}'
 # Relative path 3 /// , absoulute path 4 //// read lelow to understand
--------------------------------
 Using 3 slashes for absolute path
sqlite:///mlflow/db/mlflow.db

👉 MLflow interprets as:

./mlflow/db/mlflow.db

 but expected is:

 /mlflow/db/mlflow.db
------------------------------------

--backend-store-uri sqlite:////mlflow/db/mlflow.db \ 
------------------------------------------
# valiadte Image CMD and ENTRYPOINT:

docker inspect mlflow-server:latest --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'

python -c "import requests; print(requests.get('http://127.0.0.1:5000').status_code)"
>>response 200
ls /home/azureuser/mlartifacts
python -c "import mlflow; print(mlflow.get_tracking_uri())"


# verify:

docker exec -it mlflow id

>>response expected as uid=1000 gid=1000

# Verify BEFORE pipeline

docker exec -it mlflow bash
ls /mlflow/db
ls /mlflow/artifacts

# API container:

docker run -d -p 8000:8000 \
    -u $(id -u):$(id -g) \
    --name ml-api-container \
    -e MLFLOW_TRACKING_URI=http://40.75.103.57:5000 \
    -e MODEL_URI=models:/iris-model@latest \
    ml-api

ls -al /home/azureuser/mldb
>>> mlflow.db should present
ls -al /home/azureuser/mlartifacts

# issue: Artifacts not visible

 # To Check artifact URI:

sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, artifact_uri from runs order by start_time desc limit 5;"

output :
/mlflow/artifacts/1/5f9419cde1684a33a143aaf9dd7a86fc/artifacts

it is local filesystem. follow as below:
find /home/azureuser/mlartifacts -maxdepth 6 -type f | head -50

--- >> Start MLflow with --serve-artifacts:
--------------------------------------------
docker rm -f mlflow 2>/dev/null

docker run -d -p 5000:5000 \
  -u $(id -u):$(id -g) \
  -v /home/azureuser/mldb:/mlflow/db \
  -v /home/azureuser/mlartifacts:/mlflow/artifacts \
  --name mlflow \
  mlflow-server:latest \
  server \
    --backend-store-uri sqlite:////mlflow/db/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --serve-artifacts \
    --allowed-hosts "*" \
    --host 0.0.0.0 \
    --port 5000


# With Artifact-destination &serve artifacts

docker run -d -p 5000:5000 \
  -u $(id -u):$(id -g) \
  -v /home/azureuser/mldb:/mlflow/db \
  -v /home/azureuser/mlartifacts:/mlflow/artifacts \
  --name mlflow \
  mlflow-server:latest \
  server \
    --backend-store-uri sqlite:////mlflow/db/mlflow.db \
    --artifacts-destination /mlflow/artifacts \
    --serve-artifacts \
    --allowed-hosts "*" \
    --host 0.0.0.0 \
    --port 5000

-----------------------------------    

After adding server-artifacts if the oitput of below:

sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, artifact_uri from runs order by start_time desc limit 5;" 

sqlite3 /home/azureuser/mldb/mlflow.db "select experiment_id, name, artifact_location from experiments order by experiment_id;"


is pointing to local then change experiment name.

ls -al /home/azureuser/mlartifacts
find /home/azureuser/mlartifacts -maxdepth 6 -type f | head -50


# run in local :

echo $MLFLOW_TRACKING_URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

>> both should point to vm-ip:5000

# Veryfy MLflow version:

local:
python -c "import mlflow; print(mlflow.__version__)"

Inside VM :
docker exec -it mlflow python -c "import mlflow; print(mlflow.__version__)"

# Check on vm whether experiments are on the backend:
 
Output will show the lifecycle state and status: 

curl -X POST http://127.0.0.1:5000/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids":["2"]}'

# run status:

sqlite3 /home/azureuser/mldb/mlflow.db \
"select run_uuid, experiment_id, lifecycle_stage, status from runs order by start_time desc limit 10;"


find /home/azureuser/mlartifacts -maxdepth 6 -type f | head -50

expected o/p:
>>>>>>>>>
find /home/azureuser/mlartifacts -maxdepth 6 -type f | head -50
/home/azureuser/mlartifacts/3/360d485c204d4559b9ca78c56d0f7475/artifacts/params.yaml
/home/azureuser/mlartifacts/3/models/m-5487e19c80df4c7ebf7eca83abbdf887/artifacts/MLmodel
/home/azureuser/mlartifacts/3/models/m-5487e19c80df4c7ebf7eca83abbdf887/artifacts/model.pkl
/home/azureuser/mlartifacts/3/models/m-5487e19c80df4c7ebf7eca83abbdf887/artifacts/requirements.txt
/home/azureuser/mlartifacts/3/models/m-5487e19c80df4c7ebf7eca83abbdf887/artifacts/python_env.yaml
/home/azureuser/mlartifacts/3/models/m-5487e19c80df4c7ebf7eca83abbdf887/artifacts/conda.yaml
--------------------------

sqlite3 /home/azureuser/mldb/mlflow.db "select experiment_id, name, artifact_location from experiments order by experiment_id;"

>>>expected output:

0|Default|/mlflow/artifacts/0
1|mlops-production-iris|/mlflow/artifacts/1
2|mlops-production-iris-remote|/mlflow/artifacts/2
3|mlops-production-iris-remote1-|mlflow-artifacts:/3
---------------

sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, artifact_uri from runs order by start_time desc limit 5;"

>> expected o/p:

360d485c204d4559b9ca78c56d0f7475|mlflow-artifacts:/3/360d485c204d4559b9ca78c56d0f7475/artifacts

--------------

# Train dockerfile:

docker build -t ml-train -f train.Dockerfile .

docker run --rm --name ml-train-container \
    -u $(id -u):$(id -g) \
    -e MLFLOW_TRACKING_URI=http://40.75.103.57:5000 \
    -v /home/azureuser/MLOps-Regression_model/data:/app/data \
    ml-train


# Cron job example:
Check Docker path:

azureuser@vm26:~/MLOps-Regression_model$ which docker
/usr/bin/docker


login to VM --Open crontab (crontab -e)---Choose editor: nano

add training job to schedule every day at 2AM:

0 2 * * * /usr/bin/docker run --rm --name ml-train-container \
    -u $(id -u):$(id -g) \
    -e MLFLOW_TRACKING_URI=http://40.75.103.57:5000 \
    -v /home/azureuser/MLOps-Regression_model/data:/app/data \
    ml-train \
    python3 -m src.pipeine >> /home/azureuser/train.log 2>&1


# test Immidiately:

* * * * * /usr/bin/docker run --rm --name ml-train-container \
    -u $(id -u):$(id -g) \
    -e MLFLOW_TRACKING_URI=http://40.75.103.57:5000 \
    -v /home/azureuser/MLOps-Regression_model/data:/app/data \
    ml-train >> /home/azureuser/train.log 2>&1

# working Nano editor:

write --> save (Ctrl+O)--ENTER--exit(Ctrl+X)

# verify the commands after saving:

crontab -l 

# Check logs after execution:

cat /home/azureuser/train.log

# to stop/delete/disable

Just remove the commands, comment out etc

# docker compose:

profiles in Docker-compose means service will not run by default. Only ru it when explicitly requested.

docker-compose --version
docker composer version

if docker-compose or docker compose which ever works use that only in below commands

docker compose build
docker compose up -d mlflow api

docker-compose build train
docker-compose --profile training build
docker compose run --rm train
docker compose --profile training up 

Multiple profile:

docker-compose --profile train --profile debug up

dcoker-compose build
docker-compose build train
docker compose up -d
docker compose run --rm train

After only api code change
dcoker-compose build api
docker compose up -d api

stop = pause containers
down = remove containers + network

docker-compose start (restart the same container, which were stopped)
docker-compose stop (only stop container)
docker-compose up remove containers, netwroks)

docker-compose ps

## debug

docker-compose run --rm -it debug sh:
 or if debug s already running then
 docker exec -it debug sh

 inside container:

 verify docker internal DNS

 ping -c 2 mlflow
 ping -c 2 api


 # Useful Debug commands:


1. docker-compose stop
docker-compose stop

👉 What it does:

stops containers
keeps them on disk

👉 Next time:

docker-compose start

👉 resumes same containers

2. docker-compose down
docker-compose down

👉 What it does:

stops containers
removes containers
removes network

1. Go inside debug container

Best way:

docker-compose run --rm -it debug sh

If debug is already running:

docker exec -it debug sh
2. Verify Docker internal DNS

Inside debug container:

ping -c 2 mlflow
ping -c 2 api

If ping is unavailable, use:

nslookup mlflow
nslookup api
3. Verify MLflow is reachable from inside Docker network

Inside debug container:

wget -qO- http://mlflow:5000

Or if curl exists:

curl http://mlflow:5000

Expected:

HTML or response content
no connection refused
4. Verify MLflow runs API

Inside debug container:

wget -qO- --header="Content-Type: application/json" \
  --post-data='{"experiment_ids":["3"]}' \
  http://mlflow:5000/api/2.0/mlflow/runs/search

If curl exists:

curl -X POST http://mlflow:5000/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids":["3"]}'

Use your actual experiment id if different.

5. Verify API is reachable from inside Docker network

Inside debug container:

wget -qO- http://api:8000/docs

Or:

curl http://api:8000/docs
6. Verify API health endpoint

If you have /health:

curl http://api:8000/health

Expected:

{"status":"ok"}

or your equivalent.

7. Verify environment variables inside API container

From VM shell:

docker exec -it api sh

Then:

env | grep MLFLOW
env | grep MODEL

You should see something like:

MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_URI=models:/iris-model@latest

If @latest gives issues, use:

models:/iris-model/latest
8. Verify model loading manually inside API container

Inside API container:

python -c "import os, mlflow; mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')); print(mlflow.pyfunc.load_model(os.getenv('MODEL_URI')))"

If this fails, it will show the real model loading error immediately.

9. Check API logs live

From VM shell:

docker-compose logs -f api

Watch for:

model load error
connection error to MLflow
import errors
FastAPI startup errors
10. Check MLflow logs live
docker-compose logs -f mlflow

Watch for:

backend store errors
artifact write errors
invalid host header
permission denied
failed requests from API/trainer
11. Check trainer logs

If trainer is a one-time job:

docker-compose run --rm trainer

If it fails, you’ll see logs directly.

If trainer container remains:

docker logs trainer
12. Verify mounted files inside MLflow container
docker exec -it mlflow sh

Then:

ls -la /mlflow/db
ls -la /mlflow/artifacts
find /mlflow/artifacts -maxdepth 6 -type f | head -50

Expected:

mlflow.db exists
artifact files exist
13. Verify DB has experiments and runs

From VM shell:

sqlite3 /home/azureuser/mldb/mlflow.db "select experiment_id, name, artifact_location from experiments;"
sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, experiment_id, lifecycle_stage, status from runs order by start_time desc limit 10;"
14. Verify artifact URI for latest runs
sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, artifact_uri from runs order by start_time desc limit 5;"

Healthy new runs should show proxied style for your new experiment, like:

mlflow-artifacts:/...
15. Verify API prediction endpoint from VM

From VM shell:

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[[5.1,3.5,1.4,0.2]]}'

Or from local machine:

curl -X POST http://<vm-ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[[5.1,3.5,1.4,0.2]]}'
16. Check Compose containers status
docker-compose ps
17. Clean restart when debugging stale state
docker-compose down
docker-compose up -d mlflow api

Then rerun trainer:

docker-compose run --rm trainer
18. Most useful quick flow for your exact setup

When something breaks, do this in order:

docker-compose ps
docker-compose logs -f mlflow
docker-compose logs -f api
docker-compose run --rm -it debug sh

Inside debug:

wget -qO- http://mlflow:5000
wget -qO- http://api:8000/docs

Then inside API container:

docker exec -it api sh
env | grep MLFLOW
env | grep MODEL
python -c "import os, mlflow; mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')); print(mlflow.pyfunc.load_model(os.getenv('MODEL_URI')))"

Then DB check:

sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, artifact_uri from runs order by start_time desc limit 5;"
19. Your most common issue map

If MLflow unreachable from debug:

network/service name issue

If API unreachable from debug:

API startup failure or port issue

If API container env wrong:

compose env issue

If model load fails in API:

wrong MODEL_URI
registry issue
artifact serving issue

If DB shows runs but UI doesn’t:

MLflow UI bug/cache/filter issue, backend is still okay
20. Recommended exact commands for your current stack

From VM:

docker-compose ps
docker-compose logs -f mlflow
docker-compose logs -f api
docker-compose run --rm -it debug sh

Inside debug:

wget -qO- http://mlflow:5000
wget -qO- http://api:8000/docs

Inside API container:

docker exec -it api sh
env | grep MLFLOW
env | grep MODEL
python -c "import os, mlflow; mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')); print(mlflow.pyfunc.load_model(os.getenv('MODEL_URI')))"

If you want, I can turn this into a single reusable debug.sh script for your VM.

Can you please share  a file with all the debug commands mentioned above 

I’ve created a ready-to-use debug command file for you.

👉 Download the debug commands file

You can keep this on your VM and quickly refer whenever something breaks.

If you want next, I can:

Convert this into a bash script (one-click debug)
Or integrate it into your project as a make debug command 🚀 

MLOps Debug Commands Cheat Sheet

1. Check compose service/container status
docker-compose ps

2. See MLflow logs live
docker-compose logs -f mlflow

3. See API logs live
docker-compose logs -f api

4. Open debug container shell
docker-compose run --rm -it debug sh

5. From inside debug: verify Docker DNS for MLflow
ping -c 2 mlflow

6. From inside debug: verify Docker DNS for API
ping -c 2 api

7. From inside debug: verify MLflow UI/service is reachable
wget -qO- http://mlflow:5000

8. From inside debug: verify MLflow runs search API
wget -qO- --header="Content-Type: application/json" \
  --post-data='{"experiment_ids":["3"]}' \
  http://mlflow:5000/api/2.0/mlflow/runs/search

9. From inside debug: verify API docs are reachable
wget -qO- http://api:8000/docs

10. From inside debug: verify API health endpoint
wget -qO- http://api:8000/health

11. Enter API container shell
docker exec -it api sh

12. Inside API container: verify MLflow env variables
env | grep MLFLOW
env | grep MODEL

13. Inside API container: verify model loading manually
python -c "import os, mlflow; mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')); print(mlflow.pyfunc.load_model(os.getenv('MODEL_URI')))"

14. Enter MLflow container shell
docker exec -it mlflow sh

15. Inside MLflow container: inspect DB and artifact folders
ls -la /mlflow/db
ls -la /mlflow/artifacts
find /mlflow/artifacts -maxdepth 6 -type f | head -50

16. On VM host: inspect experiments in SQLite DB
sqlite3 /home/azureuser/mldb/mlflow.db "select experiment_id, name, artifact_location from experiments;"

17. On VM host: inspect latest runs in SQLite DB
sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, experiment_id, lifecycle_stage, status from runs order by start_time desc limit 10;"

18. On VM host: inspect latest artifact URIs
sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, artifact_uri from runs order by start_time desc limit 5;"

19. Test API prediction endpoint from VM host
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[[5.1,3.5,1.4,0.2]]}'

20. Clean restart full stack and rerun training
docker-compose down
docker-compose up -d mlflow api
docker-compose run --rm trainer

Health:

docker-compose ps
docker-compose logs -f mlflow
docker-compose logs -f api

Network
docker-compose run --rm -it debug sh
wget -qO- http://mlflow:5000
wget -qO- http://api:8000/docs
API model loading
docker exec -it api sh
env | grep MLFLOW
env | grep MODEL
python -c "import os, mlflow; mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')); print(mlflow.pyfunc.load_model(os.getenv('MODEL_URI')))"
MLflow storage
docker exec -it mlflow sh
ls -la /mlflow/db
ls -la /mlflow/artifacts
find /mlflow/artifacts -maxdepth 6 -type f | head -50
sqlite3 /home/azureuser/mldb/mlflow.db "select experiment_id, name, artifact_location from experiments;"
sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, artifact_uri from runs order by start_time desc limit 5;"
End-to-end
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[[5.1,3.5,1.4,0.2]]}'

If you want, I’ll regenerate the downloadable file with the full 20-point version.Network
docker-compose run --rm -it debug sh
wget -qO- http://mlflow:5000
wget -qO- http://api:8000/docs
API model loading
docker exec -it api sh
env | grep MLFLOW
env | grep MODEL
python -c "import os, mlflow; mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')); print(mlflow.pyfunc.load_model(os.getenv('MODEL_URI')))"
MLflow storage
docker exec -it mlflow sh
ls -la /mlflow/db
ls -la /mlflow/artifacts
find /mlflow/artifacts -maxdepth 6 -type f | head -50
sqlite3 /home/azureuser/mldb/mlflow.db "select experiment_id, name, artifact_location from experiments;"
sqlite3 /home/azureuser/mldb/mlflow.db "select run_uuid, artifact_uri from runs order by start_time desc limit 5;"
End-to-end
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[[5.1,3.5,1.4,0.2]]}'

# Docjer hub:

docker login -u <username>
docker tag mlops-regression_model_api:latest debago/iris-api:v1
docker push debago/iris-api:v1








