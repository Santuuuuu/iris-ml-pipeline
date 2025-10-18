#!/bin/bash
echo "=== LOCAL CI TEST ==="

# Test Python installation
echo "1. Testing Python..."
python --version
python -c "import sys; print('Python path:', sys.executable)"

# Test basic imports
echo "2. Testing basic imports..."
python -c "
try:
    import pandas as pd
    print('✅ pandas OK')
except ImportError as e:
    print('❌ pandas failed:', e)

try:
    import sklearn
    print('✅ sklearn OK') 
except ImportError as e:
    print('❌ sklearn failed:', e)

try:
    import pytest
    print('✅ pytest OK')
except ImportError as e:
    print('❌ pytest failed:', e)
"

# Test data creation
echo "3. Testing data creation..."
python -c "
try:
    from sklearn.datasets import load_iris
    import pandas as pd
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df.to_csv('data/iris.csv', index=False)
    print('✅ Data creation OK')
except Exception as e:
    print('❌ Data creation failed:', e)
"

# Test running pytest
echo "4. Testing pytest..."
python -m pytest tests/test_always_passes.py -v 2>&1 | head -20

echo "=== LOCAL TEST COMPLETE ==="
