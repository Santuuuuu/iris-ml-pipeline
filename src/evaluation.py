import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import json

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate model performance"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        
        for i, (prec, rec) in enumerate(zip(precision_per_class, recall_per_class)):
            metrics[f'precision_class_{i}'] = prec
            metrics[f'recall_class_{i}'] = rec
        
        self.metrics = metrics
        return metrics
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report"""
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'summary_metrics': self.metrics
        }
    
    def save_metrics(self, path: str = 'evaluation_metrics.json'):
        """Save evaluation metrics to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
