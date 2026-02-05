"""
Test RoleAwareContext with actual production merged parquet files.

Verifies:
1. Quantile columns (mu, sigma) are properly detected and loaded
2. quantile_z is computed correctly from production data
3. CBOE overlay columns are accessible
4. Date-based lookups work with proper pd.Timestamp objects
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.portfolio.role_aware_context import RoleAwareContext


class TestRoleAwareContextProduction(unittest.TestCase):
    """Test RoleAwareContext against actual merged parquet files."""
    
    @classmethod
    def setUpClass(cls):
        """Load production data once for all tests."""
        cls.symbol = 'AAPL'
        cls.horizon = 63
        cls.parquet_path = Path('cache/merged') / cls.symbol / f'{cls.symbol}_h{cls.horizon}_merged_portfolio.parquet'
        
        # Skip tests if production data not available
        if not cls.parquet_path.exists():
            raise unittest.SkipTest(f"Production parquet not found: {cls.parquet_path}")
        
        # Load parquet directly
        cls.df = pd.read_parquet(cls.parquet_path)
        
        # Create context
        cls.ctx = RoleAwareContext(
            symbols=[cls.symbol],
            horizon=cls.horizon,
            parquet_dir=Path('cache/merged'),
            data_source='portfolio'
        )
        
        # Find test date with valid quantile data
        sigma_col = cls.ctx._quantile_sigma_col.get(cls.symbol)
        if sigma_col not in cls.df.columns:
            raise unittest.SkipTest(f"Sigma column {sigma_col} not in parquet")
        
        valid_mask = cls.df[sigma_col] > 1e-9
        cls.valid_dates = cls.df[valid_mask]['date']
        
        if len(cls.valid_dates) == 0:
            raise unittest.SkipTest("No dates with valid quantile data")
        
        # Use middle date for testing
        cls.test_date = cls.valid_dates.iloc[len(cls.valid_dates) // 2]
        cls.test_row = cls.df[cls.df['date'] == cls.test_date].index[0]
    
    def test_quantile_columns_detected(self):
        """Test that quantile mu and sigma columns are properly detected."""
        mu_col = self.ctx._quantile_mu_col.get(self.symbol)
        sigma_col = self.ctx._quantile_sigma_col.get(self.symbol)
        
        self.assertIsNotNone(mu_col, "Quantile mu column not detected")
        self.assertIsNotNone(sigma_col, "Quantile sigma column not detected")
        
        # Verify columns exist in parquet
        self.assertIn(mu_col, self.df.columns, f"Mu column {mu_col} not in parquet")
        self.assertIn(sigma_col, self.df.columns, f"Sigma column {sigma_col} not in parquet")
        
        # Verify columns are loaded into feature matrix
        feat_cols = self.ctx._feat_cols.get(self.symbol, [])
        self.assertIn(mu_col, feat_cols, f"Mu column {mu_col} not loaded")
        self.assertIn(sigma_col, feat_cols, f"Sigma column {sigma_col} not loaded")
    
    def test_quantile_z_computation(self):
        """Test that quantile_z is computed correctly from production data."""
        # Get expected values from parquet
        mu_col = self.ctx._quantile_mu_col.get(self.symbol)
        sigma_col = self.ctx._quantile_sigma_col.get(self.symbol)
        
        mu_val = self.df.loc[self.test_row, mu_col]
        sigma_val = self.df.loc[self.test_row, sigma_col]
        expected_z = mu_val / sigma_val
        
        # Get context result - must use pd.Timestamp
        day_ctx = self.ctx.get_for_day(self.test_date)
        actual_z = day_ctx.quantile_z[0]
        
        # Verify
        self.assertGreater(abs(expected_z), 0.1, "Expected z-score should be non-zero")
        self.assertAlmostEqual(actual_z, expected_z, places=4,
                              msg=f"quantile_z mismatch: expected {expected_z:.4f}, got {actual_z:.4f}")
    
    def test_date_lookup_with_timestamp(self):
        """Test that date lookups work with pd.Timestamp objects."""
        # This should work without errors
        day_ctx = self.ctx.get_for_day(self.test_date)
        
        self.assertIsNotNone(day_ctx, "Context should be returned for valid date")
        self.assertIsNotNone(day_ctx.quantile_z, "quantile_z should be populated")
        self.assertEqual(len(day_ctx.quantile_z), 1, "quantile_z should have 1 element")
    
    def test_cboe_columns_loaded(self):
        """Test that CBOE overlay columns are accessible."""
        feat_cols = self.ctx._feat_cols.get(self.symbol, [])
        
        # Check for some key CBOE columns
        cboe_cols = [c for c in feat_cols if 'cboe_term_' in c]
        
        self.assertGreater(len(cboe_cols), 0, "Should have CBOE columns loaded")
        
        # Verify specific important columns
        expected_cboe_cols = [
            'cboe_term_vxst_vix_term_slope',
            'cboe_term_panic_premium',
            'cboe_term_vol_risk_premium',
        ]
        
        for col in expected_cboe_cols:
            if col in self.df.columns:  # Only check if in parquet
                self.assertIn(col, feat_cols, f"Expected CBOE column {col} not loaded")
    
    def test_feature_matrix_alignment(self):
        """Test that feature matrix rows align with parquet dates."""
        # Get feature matrix
        feat_mat = self.ctx._feat_mat.get(self.symbol)
        
        # Get row mapping
        row_by_date = self.ctx._row_by_date.get(self.symbol, {})
        
        # Verify test date is in mapping
        test_date_int = int(str(self.test_date)[:10].replace('-', ''))
        self.assertIn(test_date_int, row_by_date, f"Test date {test_date_int} not in row mapping")
        
        # Get row index
        row_idx = row_by_date[test_date_int]
        
        # Verify matrix row matches parquet row
        mu_col = self.ctx._quantile_mu_col.get(self.symbol)
        feat_cols = self.ctx._feat_cols.get(self.symbol, [])
        mu_idx = feat_cols.index(mu_col)
        
        mu_from_matrix = feat_mat[row_idx, mu_idx]
        mu_from_parquet = self.df.loc[self.test_row, mu_col]
        
        self.assertAlmostEqual(mu_from_matrix, mu_from_parquet, places=6,
                              msg=f"Matrix value {mu_from_matrix} != parquet value {mu_from_parquet}")
    
    def test_column_loading_percentage(self):
        """Test that column loading is selective (not all columns needed)."""
        total_cols = len(self.df.columns)
        loaded_cols = len(self.ctx._feat_cols.get(self.symbol, []))
        
        loading_pct = 100 * loaded_cols / total_cols
        
        # Should load a reasonable subset (not all columns)
        self.assertGreater(loading_pct, 1, "Should load at least 1% of columns")
        self.assertLess(loading_pct, 50, "Should load less than 50% (selective loading)")
        
        # Should load at least the critical columns
        self.assertGreater(loaded_cols, 10, "Should load more than 10 columns")


if __name__ == '__main__':
    unittest.main()
