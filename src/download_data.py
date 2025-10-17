import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

def download_iris_data():
    """Download IRIS dataset and save to data/ folder"""
    # Load iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Rename columns to match our expected format
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/iris.csv', index=False)
    print(f"Downloaded IRIS dataset with {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Species distribution:\n{df['species'].value_counts()}")

if __name__ == "__main__":
    download_iris_data()
