# MLOps-Regression_model
End-to-End MLOps system (FastAPI +MLflow+Docker + Terrafom

## Project Structure:

mlops-project/
│
├── src/
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
