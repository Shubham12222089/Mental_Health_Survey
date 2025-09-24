import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
import os

def train_model():
    """Simple model training function with MLflow tracking"""
    
    # Create Models directory
    os.makedirs("Models", exist_ok=True)
    
    # Set MLflow tracking URI to local file storage
    mlflow.set_tracking_uri("file:///./mlruns")
    
    # Start MLflow run
    with mlflow.start_run():
        # Load processed data
        print("Loading processed data...")
        df = pd.read_csv("data/processed/survey_data.csv")
        
        # Prepare features and target
        X = df.drop(columns=["treatment"])
        y = df["treatment"]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Train set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Model parameters
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Log parameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", X_train.shape[0])
        
        # Train model
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        
        # Save model locally
        model_path = "Models/model.pkl"
        joblib.dump(model, model_path)
        
        # Log model to MLflow with input example to avoid warnings
        try:
            mlflow.sklearn.log_model(
                model, 
                "random_forest_model",
                input_example=X_train[:5]  # Add input example to avoid warning
            )
            print("Model logged to MLflow successfully")
        except Exception as e:
            print(f"Warning: Could not log model to MLflow: {e}")
            print("Model still saved locally to Models/model.pkl")
        
        print("=== Training Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")
        
        # Show classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model

if __name__ == "__main__":
    train_model()