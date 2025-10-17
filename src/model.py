import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

class IrisModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess the IRIS dataset"""
        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = df['species']
        
        # Convert species to numerical labels
        species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
        y_encoded = y.map(species_mapping)
        
        return X, y_encoded, species_mapping
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """Train the model on IRIS data"""
        X, y, mapping = self.preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
    
    def predict(self, features: np.ndarray):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(features)
    
    def save_model(self, path: str = 'models/iris_model.joblib'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    def load_model(self, path: str = 'models/iris_model.joblib'):
        """Load trained model"""
        self.model = joblib.load(path)
        self.is_trained = True
