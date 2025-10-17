import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
import yaml

def load_parameters():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_model():
    """Train model and save metrics"""
    # Load parameters
    params = load_parameters()
    train_params = params['train']
    
    # Load data
    df = pd.read_csv('data/iris.csv')
    
    # Prepare features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    
    # Encode target
    species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    y_encoded = y.map(species_mapping)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=train_params['test_size'], 
        random_state=train_params['random_state'],
        stratify=y_encoded
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=train_params['n_estimators'],
        random_state=train_params['random_state']
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/iris_model.joblib')
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'dataset_size': len(df),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': X.columns.tolist()
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Model training completed!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: models/iris_model.joblib")
    print(f"Metrics saved to: metrics.json")

if __name__ == "__main__":
    train_model()
