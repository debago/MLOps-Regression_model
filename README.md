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
├── app/
│   ├── main.py          # FastAPI app
│   ├── model_loader.py  # Load MLflow model
│
├── mlruns/              # MLflow tracking (auto)
│
├── Dockerfile
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