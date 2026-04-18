from src.drift_detection import run_drift_detection
import subprocess
import sys


def trigger():
    drift = run_drift_detection()

    if drift:
        print("🚨 Drift detected → retraining started")

        subprocess.run([sys.executable, "-m", "src.pipeline"])

    else:
        print("✅ No drift → no retraining")


if __name__ == "__main__":
    trigger()