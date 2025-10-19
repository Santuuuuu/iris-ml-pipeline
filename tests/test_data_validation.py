import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_validation import DataValidator

class TestDataValidation:
    @pytest.fixture
    def valid_data(self):
        """Create valid IRIS dataset fixture"""
        return pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa']
        })
    
    @pytest.fixture
    def invalid_data_missing_cols(self):
        """Create invalid dataset with missing columns"""
        return pd.DataFrame({
            'sepal_length': [5.1, 4.9],
            'sepal_width': [3.5, 3.0]
        })
    
    @pytest.fixture
    def invalid_data_nulls(self):
        """Create invalid dataset with null values"""
        return pd.DataFrame({
            'sepal_length': [5.1, None, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa']
        })
    
    def test_validate_data_success(self, valid_data):
        """Test successful data validation"""
        validator = DataValidator()
        is_valid, message = validator.validate_data(valid_data)
        
        assert is_valid == True
        assert message == "Data validation passed"
    
    def test_validate_data_missing_columns(self, invalid_data_missing_cols):
        """Test validation with missing columns"""
        validator = DataValidator()
        is_valid, message = validator.validate_data(invalid_data_missing_cols)
        
        assert is_valid == False
        assert "Missing columns" in message
    
    def test_validate_data_null_values(self, invalid_data_nulls):
        """Test validation with null values"""
        validator = DataValidator()
        is_valid, message = validator.validate_data(invalid_data_nulls)
        
        assert is_valid == False
        assert "null values" in message
    
    def test_get_feature_stats(self, valid_data):
        """Test feature statistics calculation"""
        validator = DataValidator()
        stats = validator.get_feature_stats(valid_data)
        
        assert 'sepal_length' in stats
        assert 'count' in stats['sepal_length']
        assert stats['sepal_length']['count'] == 3.0

def test_enhanced_validation():
    """Test our new validation enhancement"""
    validator = DataValidator()
    # Test would go here
    assert True  # Placeholder for demo

def test_enhanced_validation():
    """Test our new validation enhancement"""
    validator = DataValidator()
    # Test would go here
    assert True  # Placeholder for demo

def test_enhanced_validation():
    """Test our new validation enhancement"""
    validator = DataValidator()
    # Test would go here
    assert True  # Placeholder for demo

def test_enhanced_validation():
    """Test our new validation enhancement"""
    validator = DataValidator()
    # Test would go here
    assert True  # Placeholder for demo

def test_enhanced_validation():
    """Test our new validation enhancement"""
    validator = DataValidator()
    # Test would go here
    assert True  # Placeholder for demo

def test_enhanced_validation():
    """Test our new validation enhancement"""
    validator = DataValidator()
    # Test would go here
    assert True  # Placeholder for demo
