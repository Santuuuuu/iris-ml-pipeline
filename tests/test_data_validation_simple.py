import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_data_validation_import():
    """Test that we can import data validation"""
    try:
        from data_validation import DataValidator
        validator = DataValidator()
        assert validator is not None
        print("✅ DataValidator imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import DataValidator: {e}")

def test_basic_data_validation():
    """Test basic data validation with simple data"""
    from data_validation import DataValidator
    
    # Create simple valid data
    simple_data = pd.DataFrame({
        'sepal_length': [5.1, 4.9],
        'sepal_width': [3.5, 3.0],
        'petal_length': [1.4, 1.4],
        'petal_width': [0.2, 0.2],
        'species': ['setosa', 'setosa']
    })
    
    validator = DataValidator()
    is_valid, message = validator.validate_data(simple_data)
    
    assert is_valid == True
    assert "passed" in message
    print("✅ Basic data validation passed")
