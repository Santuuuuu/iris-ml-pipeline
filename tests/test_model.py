import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import IrisModel

class TestIrisModel:
    @pytest.fixture
    def sample_data(self):
        """Create sample IRIS data for testing"""
        return pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 7.0, 6.4, 6.9],
            'sepal_width': [3.5, 3.0, 3.2, 3.2, 3.2, 3.1],
            'petal_length': [1.4, 1.4, 1.3, 4.7, 4.5, 4.9],
            'petal_width': [0.2, 0.2, 0.2, 1.4, 1.5, 1.5],
            'species': ['setosa', 'setosa', 'setosa', 'virginica', 'virginica', 'virginica']
        })
    
    @pytest.fixture
    def model(self):
        return IrisModel()
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.is_trained == False
        assert hasattr(model, 'model')
    
    def test_preprocess_data(self, model, sample_data):
        """Test data preprocessing"""
        X, y, mapping = model.preprocess_data(sample_data)
        
        assert X.shape[1] == 4  # 4 features
        assert len(y) == len(sample_data)
        assert set(mapping.keys()) == {'setosa', 'versicolor', 'virginica'}
    
    def test_model_training(self, model, sample_data):
        """Test model training"""
        results = model.train(sample_data, test_size=0.3)
        
        assert model.is_trained == True
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        assert 'feature_importance' in results
        assert all(0 <= acc <= 1 for acc in [results['train_accuracy'], results['test_accuracy']])
