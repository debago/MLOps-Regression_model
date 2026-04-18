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

def save_reference_data(X_train):
    os.makedirs("data/reference", exist_ok=True)
    
    ref_df = X_train.copy()
    ref_df.to_csv("data/reference/train_reference.csv", index=False)
    print("✅ Reference dataset saved")

def save_current_data(X_test):
    os.makedirs("data/current", exist_ok=True)
    current_df = X_test.copy()
    current_df.to_csv("data/current/current_batch.csv", index=False)
    print("✅ current_batch.csv created")

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
    save_reference_data(X_train)

    # Drift Simulation - We will use this later for drift detection, but let's create the file here for now. We will overwrite it with real data in the future.
    # X_test_drifted = X_test.copy()
    # X_test_drifted["sepal length (cm)"] *= 700

    # save_current_data(X_test_drifted)
    save_current_data(X_test)

    # Model
    model = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        random_state=42
    )

    with mlflow.start_run(run_name="rf-training") as run:

        #Train
        model.fit(X_train, y_train)

        #Call Evaluate
        metrics = evaluate_model(model,X_test, y_test)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            

        # Log params
        mlflow.log_param("n_estimators", params["model"]["n_estimators"])
        mlflow.log_artifact("params.yaml")


        # Log model to and artifact store, then register to model registry

#         mlflow.sklearn.log_model(model, name="model")

#         mlflow.register_model(
#             model_uri=f"runs:/{run.info.run_id}/model",
#             name=params["mlflow"]["registered_model_name"]
# )

        # # Log model to MLflow and register
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=params["mlflow"]["registered_model_name"]
                    )
        print("Artifact URI: " + mlflow.get_artifact_uri())
        print("tracking URI: " + mlflow.get_tracking_uri())

    print("✅ Training + Evaluation + Logging complete")
    


# if __name__ == "__main__":
#     train_model()