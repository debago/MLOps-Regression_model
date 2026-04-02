import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import yaml


import os
import yaml


def load_params():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params_path = os.path.join(base_dir, "params.yaml")

    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_data():
    params = load_params()

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )

    return X_train, X_test, y_train, y_test


# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = preprocess_data()
#     print("✅ Data preprocessed")