import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

def evaluate_model():
    """Simple model evaluation function"""
    
    # Create metrics directory
    os.makedirs("metrics", exist_ok=True)
    
    # Load test data and model
    print("Loading test data and model...")
    test_data = pd.read_csv("data/processed/test.csv")
    model = joblib.load("Models/model.pkl")
    
    # Prepare test data
    X_test = test_data.drop(columns=["treatment"])
    y_test = test_data["treatment"]
    
    print(f"Evaluating model on {len(X_test)} test samples...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=== Model Evaluation Results ===")
    print(f"Test Samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Show classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    classes = ['No Treatment', 'Treatment']
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('metrics/confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix saved to metrics/confusion_matrix.png")
    
    return accuracy

if __name__ == "__main__":
    evaluate_model()