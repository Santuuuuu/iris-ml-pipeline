import pytest
import numpy as np
import json
import tempfile
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import ModelEvaluator

class TestModelEvaluation:
    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator()
    
    @pytest.fixture
    def perfect_predictions(self):
        """Perfect predictions fixture"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        return y_true, y_pred
    
    @pytest.fixture
    def imperfect_predictions(self):
        """Imperfect predictions fixture"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])  # Some errors
        return y_true, y_pred
    
    @pytest.fixture
    def binary_predictions(self):
        """Binary classification predictions"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1])  # One error
        return y_true, y_pred
    
    def test_evaluate_perfect_model(self, evaluator, perfect_predictions):
        """Test evaluation with perfect predictions"""
        y_true, y_pred = perfect_predictions
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision_macro'] == 1.0
        assert metrics['recall_macro'] == 1.0
        assert metrics['f1_macro'] == 1.0
    
    def test_evaluate_imperfect_model(self, evaluator, imperfect_predictions):
        """Test evaluation with imperfect predictions"""
        y_true, y_pred = imperfect_predictions
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        assert metrics['accuracy'] < 1.0
        assert metrics['precision_macro'] < 1.0
        assert metrics['recall_macro'] < 1.0
        assert metrics['f1_macro'] < 1.0
        
        # Specific accuracy check (2 errors out of 6 = 0.666)
        assert abs(metrics['accuracy'] - 0.666) < 0.01
    
    def test_evaluate_binary_model(self, evaluator, binary_predictions):
        """Test evaluation with binary classification"""
        y_true, y_pred = binary_predictions
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        assert metrics['accuracy'] < 1.0
        assert 'precision_class_0' in metrics
        assert 'precision_class_1' in metrics
        assert 'recall_class_0' in metrics
        assert 'recall_class_1' in metrics
    
    def test_generate_report(self, evaluator, perfect_predictions):
        """Test report generation"""
        y_true, y_pred = perfect_predictions
        report = evaluator.generate_report(y_true, y_pred)
        
        assert 'classification_report' in report
        assert 'confusion_matrix' in report
        assert 'summary_metrics' in report
        assert len(report['confusion_matrix']) == 3  # 3 classes
    
    def test_generate_report_imperfect(self, evaluator, imperfect_predictions):
        """Test report generation with imperfect predictions"""
        y_true, y_pred = imperfect_predictions
        report = evaluator.generate_report(y_true, y_pred)
        
        # Check confusion matrix structure
        cm = report['confusion_matrix']
        assert len(cm) == 3  # 3x3 matrix for 3 classes
        assert all(len(row) == 3 for row in cm)
    
    def test_save_metrics(self, evaluator, perfect_predictions, tmp_path):
        """Test saving metrics to file"""
        y_true, y_pred = perfect_predictions
        evaluator.evaluate_model(y_true, y_pred)
        
        save_path = tmp_path / "test_metrics.json"
        evaluator.save_metrics(str(save_path))
        
        # Check file exists
        assert save_path.exists()
        
        # Check file content
        with open(save_path, 'r') as f:
            saved_metrics = json.load(f)
        
        assert 'accuracy' in saved_metrics
        assert saved_metrics['accuracy'] == 1.0
    
    def test_metrics_persistence(self, evaluator, imperfect_predictions):
        """Test that metrics are persisted in the evaluator instance"""
        y_true, y_pred = imperfect_predictions
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        # Check that metrics are stored in the instance
        assert evaluator.metrics == metrics
        assert 'accuracy' in evaluator.metrics
    
    def test_edge_cases(self, evaluator):
        """Test edge cases"""
        # Single prediction
        y_true_single = np.array([0])
        y_pred_single = np.array([0])
        metrics_single = evaluator.evaluate_model(y_true_single, y_pred_single)
        assert metrics_single['accuracy'] == 1.0
        
        # All wrong predictions
        y_true_wrong = np.array([0, 1, 2])
        y_pred_wrong = np.array([1, 2, 0])  # All wrong
        metrics_wrong = evaluator.evaluate_model(y_true_wrong, y_pred_wrong)
        assert metrics_wrong['accuracy'] == 0.0
    
    def test_metric_ranges(self, evaluator, imperfect_predictions):
        """Test that all metrics are in valid ranges [0, 1]"""
        y_true, y_pred = imperfect_predictions
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        for metric_name, metric_value in metrics.items():
            if 'class' in metric_name or metric_name in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
                assert 0 <= metric_value <= 1, f"Metric {metric_name} out of range: {metric_value}"
