FROM python:3.11-slim

WORKDIR /app

# System deps if needed by some python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY train/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src ./src
COPY params.yaml .


# Make src importable
ENV PYTHONPATH=/app

# Default tracking URI can be overridden at runtime
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Run pipeline
CMD ["python3", "-m", "src.pipeline"]