import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_validation import DataValidator

class TestDataValidationComprehensive:
    """Comprehensive data validation tests"""
    
    @pytest.fixture
    def valid_iris_data(self):
        """Complete valid IRIS dataset"""
        return pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 7.0, 6.4, 6.9, 5.5, 6.5],
            'sepal_width': [3.5, 3.0, 3.2, 3.2, 3.2, 3.1, 2.6, 3.0],
            'petal_length': [1.4, 1.4, 1.3, 4.7, 4.5, 4.9, 4.4, 5.2],
            'petal_width': [0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.2, 2.0],
            'species': ['setosa', 'setosa', 'setosa', 'virginica', 'virginica', 
                       'virginica', 'versicolor', 'versicolor']
        })
    
    @pytest.fixture
    def data_with_invalid_species(self):
        """Data with invalid species names"""
        return pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'invalid_species', 'setosa']
        })
    
    @pytest.fixture
    def data_with_out_of_range_values(self):
        """Data with out-of-range numeric values"""
        return pd.DataFrame({
            'sepal_length': [5.1, 25.0, 4.7],  # 25.0 is too large
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa']
        })
    
    @pytest.fixture
    def data_with_negative_values(self):
        """Data with negative values"""
        return pd.DataFrame({
            'sepal_length': [5.1, -1.0, 4.7],  # -1.0 is invalid
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa']
        })
    
    def test_validate_complete_iris_data(self, valid_iris_data):
        """Test validation with complete IRIS dataset"""
        validator = DataValidator()
        is_valid, message = validator.validate_data(valid_iris_data)
        
        assert is_valid == True
        assert message == "Data validation passed"
    
    def test_validate_invalid_species(self, data_with_invalid_species):
        """Test validation with invalid species names"""
        validator = DataValidator()
        is_valid, message = validator.validate_data(data_with_invalid_species)
        
        assert is_valid == False
        assert "Invalid species found" in message
        assert "setosa" in message  # Should mention expected species
    
    def test_validate_out_of_range_values(self, data_with_out_of_range_values):
        """Test validation with out-of-range numeric values"""
        validator = DataValidator()
        is_valid, message = validator.validate_data(data_with_out_of_range_values)
        
        assert is_valid == False
        assert "Invalid range" in message
        assert "sepal_length" in message
    
    def test_validate_negative_values(self, data_with_negative_values):
        """Test validation with negative values"""
        validator = DataValidator()
        is_valid, message = validator.validate_data(data_with_negative_values)
        
        assert is_valid == False
        assert "Invalid range" in message
    
    def test_get_feature_stats_comprehensive(self, valid_iris_data):
        """Test comprehensive feature statistics"""
        validator = DataValidator()
        stats = validator.get_feature_stats(valid_iris_data)
        
        expected_stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        
        for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
            assert feature in stats
            for stat in expected_stats:
                assert stat in stats[feature]
        
        # Check specific values
        assert stats['sepal_length']['count'] == 8.0
        assert stats['sepal_length']['min'] == 4.7
        assert stats['sepal_length']['max'] == 7.0
    
    def test_empty_dataframe(self):
        """Test validation with empty DataFrame"""
        validator = DataValidator()
        empty_df = pd.DataFrame()
        
        is_valid, message = validator.validate_data(empty_df)
        
        assert is_valid == False
        assert "Missing columns" in message
    
    def test_mixed_case_species(self):
        """Test validation with mixed case species names"""
        mixed_case_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['Setosa', 'setosa', 'SETOSA']  # Mixed case
        })
        
        validator = DataValidator()
        is_valid, message = validator.validate_data(mixed_case_data)
        
        assert is_valid == False  # Should fail due to case sensitivity
        assert "Invalid species found" in message
    
    def test_data_with_duplicates(self, valid_iris_data):
        """Test validation with duplicate rows"""
        duplicated_data = pd.concat([valid_iris_data, valid_iris_data], ignore_index=True)
        
        validator = DataValidator()
        is_valid, message = validator.validate_data(duplicated_data)
        
        # Duplicates should still pass validation
        assert is_valid == True
        assert message == "Data validation passed"
