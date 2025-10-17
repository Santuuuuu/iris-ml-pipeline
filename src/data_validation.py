import pandas as pd
import numpy as np
from typing import Tuple

class DataValidator:
    def __init__(self):
        self.expected_columns = [
            'sepal_length', 'sepal_width', 
            'petal_length', 'petal_width', 'species'
        ]
        self.expected_species = ['setosa', 'versicolor', 'virginica']
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate IRIS dataset structure and content"""
        try:
            # Check columns
            if not all(col in df.columns for col in self.expected_columns):
                return False, f"Missing columns. Expected: {self.expected_columns}"
            
            # Check for null values
            if df.isnull().any().any():
                return False, "Dataset contains null values"
            
            # Check species values
            if not all(species in self.expected_species for species in df['species'].unique()):
                return False, f"Invalid species found. Expected: {self.expected_species}"
            
            # Check numeric ranges
            numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            for col in numeric_columns:
                if df[col].min() <= 0 or df[col].max() > 20:
                    return False, f"Invalid range in column {col}"
            
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_feature_stats(self, df: pd.DataFrame) -> dict:
        """Get basic statistics for features"""
        numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        return df[numeric_columns].describe().to_dict()
