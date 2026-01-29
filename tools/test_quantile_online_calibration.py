#!/usr/bin/env python3
"""Test script for quantile_forecast → calibration → online_learning pipeline.

Runs these three sequential families with:
- 5-year timeframe (2021-01-01 to 2026-01-24)
- Separate test cache location
- AAPL h63 only
- Isolated testing environment

This allows rapid iteration and validation of the quantile-calibration-online learning
dependency chain without processing all families.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(override=False)

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test_qoc")


def setup_test_cache(base_dir: Path) -> Path:
    """Create separate test cache directory."""
    test_cache = base_dir / "test_cache_qoc"
    test_cache.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Test cache: {test_cache}")
    return test_cache


def run_quantile_calibration_online_test(
    symbol: str = "AAPL",
    horizon: int = 63,
    start_date: str = "2021-01-01",
    end_date: str = "2026-01-24",
    test_cache_dir: Path = None,
):
    """Run quantile_forecast → calibration → online_learning in sequence."""
    
    if test_cache_dir is None:
        test_cache_dir = setup_test_cache(REPO_ROOT / "cache")
    
    logger.info("="*100)
    logger.info(f"QUANTILE → CALIBRATION → ONLINE LEARNING TEST")
    logger.info("="*100)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Horizon: {horizon}")
    logger.info(f"Date range: {start_date} → {end_date}")
    logger.info(f"Cache: {test_cache_dir}")
    logger.info("="*100)
    
    # Import aggregator panel
    from src.features.aggregator_panel import build_panel
    
    # Override cache paths for testing
    original_cache_path = os.environ.get('LOCAL_CACHE_PATH')
    os.environ['LOCAL_CACHE_PATH'] = str(test_cache_dir)
    
    # Build consolidated panel with ONLY these three families
    # Using the actual family names from aggregator_panel
    test_families = [
        "quantile_forecast",  # GBMQuantileForecaster
        "calibration",        # ProbabilityCalibrationSystem  
        "online_learning",    # OnlineLearningSystem
    ]
    
    logger.info(f"\n🚀 Building panel with families: {test_families}")
    
    try:
        panel_df = build_panel(
            symbol=symbol,
            horizon=horizon,
            start=start_date,
            end=end_date,
            families=test_families,
            cache_dir=test_cache_dir,
        )
        
        if panel_df is None or panel_df.empty:
            logger.error(f"❌ Panel generation returned empty result")
            return None
        
        logger.info(f"\n✅ Panel generated successfully")
        logger.info(f"   Shape: {panel_df.shape}")
        logger.info(f"   Date range: {panel_df.index[0]} → {panel_df.index[-1]}")
        logger.info(f"   Columns: {len(panel_df.columns)}")
        
        # Analyze each family's contribution
        logger.info("\n" + "="*100)
        logger.info("FAMILY ANALYSIS")
        logger.info("="*100)
        
        for family in test_families:
            family_cols = [c for c in panel_df.columns if c.startswith(f"{family}_")]
            if family_cols:
                logger.info(f"\n{family.upper()}:")
                logger.info(f"  Columns: {len(family_cols)}")
                logger.info(f"  Sample columns: {family_cols[:10]}")
                
                # Check for has_data flag
                has_data_col = f"{family}_has_data"
                if has_data_col in panel_df.columns:
                    has_data_pct = (panel_df[has_data_col] > 0).mean() * 100
                    logger.info(f"  Data coverage: {has_data_pct:.1f}%")
                
                # Check for zeros/NaNs
                family_df = panel_df[family_cols]
                nan_pct = family_df.isna().sum().sum() / (len(family_df) * len(family_cols)) * 100
                zero_pct = (family_df == 0).sum().sum() / (len(family_df) * len(family_cols)) * 100
                logger.info(f"  NaN%: {nan_pct:.1f}%, Zero%: {zero_pct:.1f}%")
                
                # Show summary stats for first few columns
                logger.info(f"  Summary stats (first 3 cols):")
                for col in family_cols[:3]:
                    mean = panel_df[col].mean()
                    std = panel_df[col].std()
                    logger.info(f"    {col}: mean={mean:.4f}, std={std:.4f}")
            else:
                logger.warning(f"\n{family.upper()}: ❌ NO COLUMNS FOUND")
        
        # Save test output
        output_file = test_cache_dir / f"{symbol}_h{horizon}_test_qoc.parquet"
        panel_df.to_parquet(output_file)
        logger.info(f"\n💾 Saved test output: {output_file}")
        
        # Generate summary report
        report_file = test_cache_dir / f"{symbol}_h{horizon}_test_qoc_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"QUANTILE-CALIBRATION-ONLINE TEST REPORT\n")
            f.write(f"="*100 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Horizon: {horizon}\n")
            f.write(f"Date range: {start_date} → {end_date}\n")
            f.write(f"Panel shape: {panel_df.shape}\n\n")
            
            f.write(f"COLUMN BREAKDOWN:\n")
            f.write(f"-"*100 + "\n")
            for family in test_families:
                family_cols = [c for c in panel_df.columns if c.startswith(f"{family}_")]
                f.write(f"{family}: {len(family_cols)} columns\n")
            
            f.write(f"\n" + "="*100 + "\n")
            f.write(f"DETAILED COLUMN LIST:\n")
            f.write(f"="*100 + "\n")
            for col in sorted(panel_df.columns):
                f.write(f"{col}\n")
        
        logger.info(f"📊 Saved report: {report_file}")
        
        return panel_df
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return None
    
    finally:
        # Restore original cache path
        if original_cache_path:
            os.environ['LOCAL_CACHE_PATH'] = original_cache_path
        else:
            os.environ.pop('LOCAL_CACHE_PATH', None)


def main():
    parser = argparse.ArgumentParser(
        description="Test quantile_forecast → calibration → online_learning pipeline"
    )
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Symbol to test (default: AAPL)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=63,
        help="Forecast horizon (default: 63)"
    )
    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="Start date (default: 2021-01-01)"
    )
    parser.add_argument(
        "--end",
        default="2026-01-24",
        help="End date (default: 2026-01-24)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Custom cache directory (default: auto-create in cache/test_cache_qoc)"
    )
    
    args = parser.parse_args()
    
    if args.cache_dir is None:
        args.cache_dir = setup_test_cache(REPO_ROOT / "cache")
    
    logger.info("\n" + "🧪 STARTING QUANTILE-CALIBRATION-ONLINE TEST " + "\n")
    
    result = run_quantile_calibration_online_test(
        symbol=args.symbol,
        horizon=args.horizon,
        start_date=args.start,
        end_date=args.end,
        test_cache_dir=args.cache_dir,
    )
    
    if result is not None:
        logger.info("\n" + "="*100)
        logger.info("✅ TEST COMPLETED SUCCESSFULLY")
        logger.info("="*100)
        return 0
    else:
        logger.error("\n" + "="*100)
        logger.error("❌ TEST FAILED")
        logger.error("="*100)
        return 1


if __name__ == "__main__":
    sys.exit(main())
