import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris


def train():
    # -----------------------------
    # 1. Set Remote MLflow Server
    # -----------------------------
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mlops-production")

    # -----------------------------
    # 2. Load Data
    # -----------------------------
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 3. Train Model
    # -----------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # -----------------------------
    # 4. Evaluate
    # -----------------------------
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # -----------------------------
    # 5. MLflow Tracking + Registry
    # -----------------------------
    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Log + Register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="iris-model"
        )

    print(f"✅ Model trained & registered with accuracy: {acc}")


if __name__ == "__main__":
    train()