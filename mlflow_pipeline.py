# Assignment 5: MLFlow CI/CD Pipeline
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna

# MLFlow setup
MLFLOW_TRACKING_URI = "http://34.122.162.121:8100/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris-hyperparameter-tuning")

def main():
    print("🎯 Assignment 5: MLFlow CI/CD Pipeline")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
        }
        model = RandomForestClassifier(random_state=42, **params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    
    # MLFlow logging
    with mlflow.start_run(run_name="assignment5_optimized"):
        model = RandomForestClassifier(random_state=42, **best_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model", registered_model_name="iris-classifier")
        
        print(f"✅ Model trained with accuracy: {accuracy:.4f}")
        print(f"🔗 View at: {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    main()
