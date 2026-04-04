FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary curl

# Create directories
RUN mkdir -p /mlflow/artifacts

# Expose port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import requests;r=requests.POST('http://127.0.0.1:5000/api/2.0/preview/mlflow/experiments/list', json={}); exit(0 if r.status_code==200 else 1)"

# Start MLflow server
ENTRYPOINT ["mlflow"]
CMD ["server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--allowed-hosts", "*", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

