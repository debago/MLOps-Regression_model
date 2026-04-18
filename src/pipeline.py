from src.train import train_model

def run_pipeline():
    print("🚀 Starting ML Pipeline...")
    train_model()
    print("✅ Pipeline execution complete.")

    
    # X_test_drifted = X_test.copy()
    # X_test_drifted["sepal length (cm)"] *= 1.2

    # save_current_data(X_test_drifted)

if __name__ == "__main__":
    run_pipeline()
