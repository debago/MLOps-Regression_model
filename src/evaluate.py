from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def evaluate_model(model, X_test, y_test):

    """ Evaluate the model and return metrics. No MLFLOW logging here - Handled in train.py. """
    # params = load_params()

    preds = model.predict(X_test)    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted")  
    recall = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    return metrics

