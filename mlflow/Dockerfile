FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary

# Create directories
RUN mkdir -p /mlflow/artifacts /mlflow/db

# Expose port
EXPOSE 5000

# Healthcheck
HEALTHCHECK CMD python -c "import requests; exit(0 if requests.get('http://127.0.0.1:5000').status_code==200 else 1)"
# Start MLflow server
ENTRYPOINT ["mlflow"]
CMD ["server", \
     "--backend-store-uri", "sqlite:///mlflow/db/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--allow-hosts", "*", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

