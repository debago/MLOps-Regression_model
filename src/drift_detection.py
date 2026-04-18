import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE_PATH = os.path.join(BASE_DIR, "data", "reference", "train_reference.csv")
CURRENT_PATH = os.path.join(BASE_DIR, "data", "current", "current_batch.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "drift")
HTML_REPORT_PATH = os.path.join(REPORT_DIR, "data_drift_report.html")


def run_drift_detection():
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(f"Reference file not found: {REFERENCE_PATH}")

    if not os.path.exists(CURRENT_PATH):
        raise FileNotFoundError(f"Current file not found: {CURRENT_PATH}")

    reference_df = pd.read_csv(REFERENCE_PATH)
    current_df = pd.read_csv(CURRENT_PATH)

    os.makedirs(REPORT_DIR, exist_ok=True)

    report = Report([DataDriftPreset()])
    my_eval = report.run(reference_df, current_df)

    my_eval.save_html(HTML_REPORT_PATH)

    print(f"✅ Drift report saved at: {HTML_REPORT_PATH}")
    print(type(my_eval))
    print(dir(my_eval))

    report_dict = my_eval.dict()

    drift_detected = False

    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})
            if "dataset_drift" in result:
                drift_detected = result["dataset_drift"]
                break
    except Exception as e:
        print("Warning:", e)

    print(f"🚨 Drift detected: {drift_detected}")

    return drift_detected


if __name__ == "__main__":
    run_drift_detection()