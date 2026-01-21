"""
Test ML Training Framework

Tests the machine learning training framework including feature engineering,
model training, evaluation, and prediction capabilities for financial data.

Author: DCF Lab Team
Created: 2025-01-20
"""

import os
import shutil

# Import the ML framework components
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from providers.ml_framework import (
    FeatureEngineer,
    MLConfig,
    MLTrainingFramework,
    ModelFactory,
    NeuralNetworkModel,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
    get_ml_framework,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMLConfig(unittest.TestCase):
    """Test MLConfig"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = MLConfig()

        self.assertEqual(config.model_dir, "./models")
        self.assertEqual(config.cache_dir, "./cache/ml")
        self.assertTrue(config.enable_cache)
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.validation_size, 0.2)
        self.assertEqual(config.cv_folds, 5)
        self.assertEqual(config.max_features, 100)
        self.assertEqual(config.feature_selection_k, 50)
        self.assertEqual(config.scaling_method, "standard")

    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = MLConfig(
            model_dir="./custom_models",
            random_state=123,
            test_size=0.3,
            max_features=200,
            scaling_method="minmax"
        )

        self.assertEqual(config.model_dir, "./custom_models")
        self.assertEqual(config.random_state, 123)
        self.assertEqual(config.test_size, 0.3)
        self.assertEqual(config.max_features, 200)
        self.assertEqual(config.scaling_method, "minmax")


class TestFeatureEngineer(unittest.TestCase):
    """Test FeatureEngineer"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MLConfig()
        self.engineer = FeatureEngineer(self.config)

        # Create sample financial data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        n_days = len(dates)

        # Generate realistic financial data
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = [100]
        for i in range(1, n_days):
            prices.append(prices[-1] * (1 + returns[i]))

        self.sample_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, n_days),
            'pe_ratio': np.random.uniform(15, 25, n_days),
            'vix': np.random.uniform(15, 35, n_days),
            'interest_rate': np.random.uniform(2, 5, n_days)
        }, index=dates)

    def test_feature_engineering(self):
        """Test basic feature engineering"""
        features_df, target_series = self.engineer.engineer_features(
            self.sample_data, target_column='close'
        )

        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIsInstance(target_series, pd.Series)
        # Should have many features
        self.assertGreater(len(features_df.columns), 50)
        self.assertEqual(len(features_df), len(target_series))

        # Check that we have various types of features
        feature_names = features_df.columns.tolist()

        # Price features
        self.assertIn('return_1d', feature_names)
        self.assertIn('return_5d', feature_names)

        # Technical indicators
        self.assertIn('ma_20', feature_names)
        self.assertIn('rsi', feature_names)
        self.assertIn('macd', feature_names)

        # Time features
        self.assertIn('day_of_week', feature_names)
        self.assertIn('month', feature_names)

        # Statistical features
        self.assertIn('volatility_20d', feature_names)

    def test_price_features(self):
        """Test price-based feature creation"""
        features_df = pd.DataFrame(index=self.sample_data.index)
        result_df = self.engineer._add_price_features(
            features_df, self.sample_data)

        expected_features = [
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'log_return_1d', 'log_return_5d'
        ]

        for feature in expected_features:
            self.assertIn(feature, result_df.columns)

        # Test price ratios
        price_ratio_features = [
            col for col in result_df.columns if 'price_ratio' in col]
        self.assertGreater(len(price_ratio_features), 0)

    def test_technical_indicators(self):
        """Test technical indicator creation"""
        features_df = pd.DataFrame(index=self.sample_data.index)
        result_df = self.engineer._add_technical_indicators(
            features_df, self.sample_data)

        expected_indicators = [
            'ma_20',
            'ma_50',
            'ema_12',
            'ema_26',
            'macd',
            'rsi']

        for indicator in expected_indicators:
            self.assertIn(indicator, result_df.columns)

        # Check RSI is in valid range (after warming up)
        rsi_values = result_df['rsi'].dropna()
        if len(rsi_values) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in rsi_values))

    def test_time_features(self):
        """Test time-based feature creation"""
        features_df = pd.DataFrame(index=self.sample_data.index)
        result_df = self.engineer._add_time_features(
            features_df, self.sample_data)

        expected_time_features = [
            'day_of_week', 'day_of_month', 'month', 'quarter',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'
        ]

        for feature in expected_time_features:
            self.assertIn(feature, result_df.columns)

        # Check cyclical encoding
        self.assertTrue(
            all(-1 <= val <= 1 for val in result_df['day_of_week_sin']))
        self.assertTrue(
            all(-1 <= val <= 1 for val in result_df['day_of_week_cos']))

    def test_feature_engineering_minimal_data(self):
        """Test feature engineering with minimal data"""
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=pd.date_range('2023-01-01', periods=5))

        features_df, target_series = self.engineer.engineer_features(
            minimal_data, target_column='close'
        )

        # Should still work but with fewer samples due to NaN removal
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIsInstance(target_series, pd.Series)
        self.assertGreaterEqual(
            len(features_df),
            1)  # At least one valid sample


class TestModelFactory(unittest.TestCase):
    """Test ModelFactory"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MLConfig()

    def test_create_random_forest_model(self):
        """Test Random Forest model creation"""
        model = ModelFactory.create_model('random_forest', self.config)
        self.assertIsInstance(model, RandomForestModel)

    def test_create_xgboost_model(self):
        """Test XGBoost model creation"""
        model = ModelFactory.create_model('xgboost', self.config)
        self.assertIsInstance(model, XGBoostModel)

    def test_create_ridge_model(self):
        """Test Ridge model creation"""
        model = ModelFactory.create_model('ridge', self.config)
        self.assertIsInstance(model, RidgeModel)

    def test_create_neural_network_model(self):
        """Test Neural Network model creation"""
        model = ModelFactory.create_model('neural_network', self.config)
        self.assertIsInstance(model, NeuralNetworkModel)

    def test_invalid_model_type(self):
        """Test invalid model type"""
        with self.assertRaises(ValueError):
            ModelFactory.create_model('invalid_model', self.config)


class TestBaseMLModel(unittest.TestCase):
    """Test base ML model functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MLConfig()

        # Create sample training data
        np.random.seed(42)
        self.X_train = pd.DataFrame(
            np.random.randn(
                100,
                5),
            columns=[
                'feature1',
                'feature2',
                'feature3',
                'feature4',
                'feature5'])
        self.y_train = pd.Series(np.random.randn(100))

        self.X_test = pd.DataFrame(
            np.random.randn(
                20,
                5),
            columns=[
                'feature1',
                'feature2',
                'feature3',
                'feature4',
                'feature5'])

    def test_random_forest_training(self):
        """Test Random Forest model training"""
        model = RandomForestModel(self.config)

        # Test model creation
        rf_model = model.create_model()
        self.assertIsNotNone(rf_model)

        # Test hyperparameter space
        param_space = model.get_hyperparameter_space()
        self.assertIsInstance(param_space, dict)
        self.assertIn('n_estimators', param_space)

        # Test training
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.scaler)

        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

        # Test feature importance
        importance = model.get_feature_importance(
            self.X_train.columns.tolist())
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), len(self.X_train.columns))

    def test_ridge_training(self):
        """Test Ridge model training"""
        model = RidgeModel(self.config)

        # Test training and prediction
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

        # Test feature importance (coefficients)
        importance = model.get_feature_importance(
            self.X_train.columns.tolist())
        self.assertIsInstance(importance, dict)

    def test_prediction_without_training(self):
        """Test prediction without training should raise error"""
        model = RandomForestModel(self.config)

        with self.assertRaises(ValueError):
            model.predict(self.X_test)


class TestMLTrainingFramework(unittest.TestCase):
    """Test MLTrainingFramework"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MLConfig(
            model_dir=os.path.join(self.temp_dir, "models"),
            cache_dir=os.path.join(self.temp_dir, "cache"),
            hyperparameter_optimization=False,  # Disable for faster testing
            default_models=['ridge', 'random_forest'],  # Use simpler models
            cv_folds=3  # Fewer folds for faster testing
        )
        self.framework = MLTrainingFramework(self.config)

        # Create comprehensive sample data
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        np.random.seed(42)
        n_days = len(dates)

        # Generate realistic financial data
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [100]
        for i in range(1, n_days):
            prices.append(prices[-1] * (1 + returns[i]))

        self.sample_data = pd.DataFrame({
            'close': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'volume': np.random.lognormal(15, 0.5, n_days),
            'pe_ratio': np.random.uniform(15, 25, n_days),
            'vix': np.random.uniform(15, 35, n_days)
        }, index=dates)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_framework_initialization(self):
        """Test framework initialization"""
        self.assertIsInstance(self.framework.config, MLConfig)
        self.assertIsInstance(self.framework.feature_engineer, FeatureEngineer)
        self.assertTrue(os.path.exists(self.config.model_dir))
        self.assertTrue(os.path.exists(self.config.cache_dir))

    def test_factory_function(self):
        """Test factory function"""
        framework = get_ml_framework()
        self.assertIsInstance(framework, MLTrainingFramework)

        # Test with custom config
        custom_config = MLConfig(random_state=123)
        framework_custom = get_ml_framework(custom_config)
        self.assertEqual(framework_custom.config.random_state, 123)

    def test_model_training(self):
        """Test model training process"""
        results = self.framework.train_models(
            self.sample_data,
            target_column='close',
            models=['ridge']  # Use single model for faster testing
        )

        self.assertIsInstance(results, dict)
        self.assertIn('ridge', results)

        result = results['ridge']
        self.assertEqual(result.model_type, 'ridge')
        self.assertIsNotNone(result.trained_model)
        self.assertIsNotNone(result.scaler)
        self.assertGreater(len(result.feature_names), 0)

        # Check metrics
        self.assertIn('r2', result.test_score)
        self.assertIn('mse', result.test_score)
        self.assertIn('mape', result.test_score)

        # Check trained models are stored
        self.assertIn('ridge', self.framework.trained_models)

    def test_model_comparison(self):
        """Test model comparison"""
        # Train multiple models
        self.framework.train_models(
            self.sample_data,
            target_column='close',
            models=['ridge', 'random_forest']
        )

        comparison = self.framework.get_model_comparison()

        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertGreater(len(comparison), 0)

        expected_columns = [
            'Model',
            'R² Score',
            'RMSE',
            'MAE',
            'MAPE (%)',
            'Training Time (s)',
            'Features']
        for col in expected_columns:
            self.assertIn(col, comparison.columns)

        # Should be sorted by R² Score
        r2_scores = comparison['R² Score'].tolist()
        self.assertEqual(r2_scores, sorted(r2_scores, reverse=True))

    def test_prediction_generation(self):
        """Test prediction generation"""
        # Train a model first
        self.framework.train_models(
            self.sample_data,
            target_column='close',
            models=['ridge']
        )

        # Generate predictions on subset of data
        prediction_data = self.sample_data.iloc[-30:]  # Last 30 days
        prediction_result = self.framework.predict(
            prediction_data,
            model_type='ridge'
        )

        self.assertIsInstance(prediction_result, dict)
        self.assertIn('predictions', prediction_result)
        self.assertIn('model_type', prediction_result)
        self.assertIn('feature_names', prediction_result)
        self.assertIn('model_performance', prediction_result)

        predictions = prediction_result['predictions']
        self.assertEqual(prediction_result['model_type'], 'ridge')
        self.assertGreater(len(predictions), 0)

    def test_prediction_without_trained_model(self):
        """Test prediction without trained model should raise error"""
        with self.assertRaises(ValueError):
            self.framework.predict(self.sample_data, model_type='ridge')

    def test_feature_selection(self):
        """Test feature selection functionality"""
        # Create high-dimensional feature data
        features_df = pd.DataFrame(np.random.randn(100, 200))  # 200 features
        target_series = pd.Series(np.random.randn(100))

        selected_features = self.framework._select_features(
            features_df, target_series)

        self.assertIsInstance(selected_features, pd.DataFrame)
        self.assertEqual(len(selected_features.columns),
                         self.config.feature_selection_k)

    def test_data_splitting(self):
        """Test data splitting"""
        features_df, target_series = self.framework.feature_engineer.engineer_features(
            self.sample_data, 'close')

        X_train, X_test, y_train, y_test = self.framework._split_data(
            features_df, target_series)

        # Check split proportions
        total_samples = len(features_df)
        expected_test_size = int(total_samples * self.config.test_size)

        self.assertAlmostEqual(len(X_test), expected_test_size, delta=1)
        self.assertEqual(len(X_train), total_samples - len(X_test))
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))

        # For time series, test set should be later in time
        if isinstance(features_df.index, pd.DatetimeIndex):
            self.assertGreater(X_test.index.min(), X_train.index.max())

    def test_metrics_calculation(self):
        """Test metrics calculation"""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 4.9])

        metrics = self.framework._calculate_metrics(y_true, y_pred)

        expected_metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

        # R² should be close to 1 for this good prediction
        self.assertGreater(metrics['r2'], 0.9)

    def test_empty_training_data(self):
        """Test handling of empty training data"""
        empty_data = pd.DataFrame({'close': []})

        results = self.framework.train_models(
            empty_data, target_column='close')

        # Should handle gracefully
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 0)  # No models trained

    def test_missing_target_column(self):
        """Test handling of missing target column"""
        data_without_target = self.sample_data.drop('close', axis=1)

        with self.assertRaises(KeyError):
            self.framework.train_models(
                data_without_target, target_column='close')


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestMLConfig,
        TestFeatureEngineer,
        TestModelFactory,
        TestBaseMLModel,
        TestMLTrainingFramework
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*50)
    print("ML Training Framework Tests Summary")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) ==
                      0 and len(result.errors) == 0) else 1
    exit(exit_code)
