import mlflow
import mlflow.sklearn
import os
import yaml

from sklearn.ensemble import RandomForestClassifier
from src.preprocess import preprocess_data
from src.evaluate import evaluate_model

#--------------------------------------
# Load Params
#---------------------------------------

def load_params():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params_path = os.path.join(base_dir, "params.yaml")

    with open(params_path, "r") as f:
        return yaml.safe_load(f)

#--------------------------------------
# Model Train
#---------------------------------------

def train_model():
    params = load_params()

    # MLflow setup (Env override for Docker/Kubernetes)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")


    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    print(f"🔍 MLFLOW URI: {tracking_uri}")

    # Load data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Model
    model = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        random_state=42
    )

    with mlflow.start_run(run_name="rf-training"):

        #Train
        model.fit(X_train, y_train)

        #Call Evaluate
        metrics = evaluate_model(model,X_test, y_test)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log params
        mlflow.log_param("n_estimators", params["model"]["n_estimators"])

        # Log model to MLflow and register
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=params["mlflow"]["registered_model_name"]
        )
        print("Artifact URI: " + mlflow.get_artifact_uri())
        print("tracking URI: " + mlflow.get_tracking_uri())

    print("✅ Training + Evaluation + Logging complete")
    


# if __name__ == "__main__":
#     train_model()