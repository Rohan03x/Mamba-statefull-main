"""
Live Accuracy Tracking System for DCF Suite v0201

This module implements comprehensive accuracy tracking for the complete ML pipeline:
- Directional accuracy (up/down predictions)
- Hit rate within confidence intervals
- RMSE/MAE error metrics
- Rolling window evaluations (7/30/90 days)
- Integration with governance framework and A/B testing
- Live prediction logging and validation

Features:
- Real-time accuracy calculation
- Multi-horizon accuracy tracking
- Symbol-specific performance metrics
- Model comparison and A/B testing integration
- Risk-adjusted performance metrics
- Automated reporting and alerting
"""

import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path

# Fix Windows UTF-8 encoding for emoji support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 fallback
        pass

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    DIRECTIONAL = "directional"
    PRICE_TARGET = "price_target"
    CONFIDENCE_INTERVAL = "confidence_interval"
    MULTI_HORIZON = "multi_horizon"

class AccuracyMetric(Enum):
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    HIT_RATE = "hit_rate"
    RMSE = "rmse"
    MAE = "mae"
    SHARPE_RATIO = "sharpe_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"

@dataclass
class PredictionRecord:
    """Single prediction record for tracking"""
    prediction_id: str
    timestamp: datetime
    symbol: str
    model_id: str
    model_version: str
    
    # Prediction details
    prediction_type: PredictionType
    horizon_days: int
    predicted_price: Optional[float] = None
    predicted_direction: Optional[str] = None  # 'up', 'down', 'neutral'
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    confidence_level: float = 0.95
    
    # Market context
    current_price: float = 0.0
    predicted_return: Optional[float] = None
    volatility_estimate: Optional[float] = None
    
    # Features used
    features_hash: Optional[str] = None
    feature_count: int = 0
    
    # Validation (filled later)
    actual_price: Optional[float] = None
    actual_return: Optional[float] = None
    validation_date: Optional[datetime] = None
    is_validated: bool = False

@dataclass
class AccuracyResults:
    """Accuracy calculation results"""
    metric_type: AccuracyMetric
    value: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    time_window: str
    
    # Breakdown by symbol/model
    by_symbol: Dict[str, float] = field(default_factory=dict)
    by_model: Dict[str, float] = field(default_factory=dict)
    by_horizon: Dict[int, float] = field(default_factory=dict)

class LiveAccuracyTracker:
    """Comprehensive accuracy tracking system"""
    
    def __init__(self, db_path: str = "data/accuracy_tracking.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Rolling window periods
        self.window_periods = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'yearly': 365
        }
        
    def _init_database(self):
        """Initialize SQLite database for prediction tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    predicted_price REAL,
                    predicted_direction TEXT,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    confidence_level REAL,
                    current_price REAL,
                    predicted_return REAL,
                    volatility_estimate REAL,
                    features_hash TEXT,
                    feature_count INTEGER,
                    actual_price REAL,
                    actual_return REAL,
                    validation_date TEXT,
                    is_validated BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    window_period TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    metadata TEXT
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_id, model_version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_validation ON predictions(is_validated, validation_date)")
            
    def log_prediction(self, prediction: PredictionRecord) -> str:
        """Log a new prediction for tracking"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO predictions (
                    prediction_id, timestamp, symbol, model_id, model_version,
                    prediction_type, horizon_days, predicted_price, predicted_direction,
                    confidence_lower, confidence_upper, confidence_level,
                    current_price, predicted_return, volatility_estimate,
                    features_hash, feature_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.prediction_id,
                prediction.timestamp.isoformat(),
                prediction.symbol,
                prediction.model_id,
                prediction.model_version,
                prediction.prediction_type.value,
                prediction.horizon_days,
                prediction.predicted_price,
                prediction.predicted_direction,
                prediction.confidence_lower,
                prediction.confidence_upper,
                prediction.confidence_level,
                prediction.current_price,
                prediction.predicted_return,
                prediction.volatility_estimate,
                prediction.features_hash,
                prediction.feature_count
            ))
            
        logger.info(f"Logged prediction {prediction.prediction_id} for {prediction.symbol}")
        return prediction.prediction_id
    
    def validate_predictions(self, market_data: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        """Validate pending predictions against actual market data"""
        
        validation_stats = {'validated': 0, 'pending': 0, 'errors': 0}
        current_date = datetime.now().date()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get unvalidated predictions ready for validation
            cursor = conn.execute("""
                SELECT * FROM predictions 
                WHERE is_validated = FALSE 
                AND date(timestamp, '+' || horizon_days || ' days') <= date('now')
                ORDER BY timestamp ASC
            """)
            
            predictions = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for pred_row in predictions:
                pred_dict = dict(zip(columns, pred_row))
                
                try:
                    symbol = pred_dict['symbol']
                    prediction_date = datetime.fromisoformat(pred_dict['timestamp'])
                    horizon_days = pred_dict['horizon_days']
                    target_date = (prediction_date + timedelta(days=horizon_days)).date()
                    
                    # Check if we have market data for validation
                    if symbol in market_data and target_date.isoformat() in market_data[symbol]:
                        actual_price = market_data[symbol][target_date.isoformat()]
                        current_price = pred_dict['current_price']
                        
                        actual_return = (actual_price - current_price) / current_price
                        
                        # Update with validation data
                        conn.execute("""
                            UPDATE predictions 
                            SET actual_price = ?, actual_return = ?, 
                                validation_date = ?, is_validated = TRUE
                            WHERE prediction_id = ?
                        """, (
                            actual_price,
                            actual_return,
                            current_date.isoformat(),
                            pred_dict['prediction_id']
                        ))
                        
                        validation_stats['validated'] += 1
                        
                    else:
                        validation_stats['pending'] += 1
                        
                except Exception as e:
                    logger.error(f"Error validating prediction {pred_dict.get('prediction_id')}: {e}")
                    validation_stats['errors'] += 1
        
        logger.info(f"🔍 Validation completed: {validation_stats}")
        return validation_stats
    
    def calculate_accuracy_metrics(self, 
                                   window_days: int = 30,
                                   symbols: Optional[List[str]] = None,
                                   models: Optional[List[str]] = None) -> Dict[AccuracyMetric, AccuracyResults]:
        """Calculate comprehensive accuracy metrics"""
        
        cutoff_date = (datetime.now() - timedelta(days=window_days)).isoformat()
        
        # Build query filters
        where_clauses = ["is_validated = TRUE", f"validation_date >= '{cutoff_date}'"]
        if symbols:
            symbol_list = "', '".join(symbols)
            where_clauses.append(f"symbol IN ('{symbol_list}')")
        if models:
            model_list = "', '".join(models)
            where_clauses.append(f"model_id IN ('{model_list}')")
        
        where_clause = " AND ".join(where_clauses)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get validated predictions
            df = pd.read_sql_query(f"""
                SELECT * FROM predictions 
                WHERE {where_clause}
                ORDER BY validation_date DESC
            """, conn)
            
        if df.empty:
            logger.warning(f"No validated predictions found for window {window_days} days")
            return {}
        
        results = {}
        window_name = f"{window_days}d"
        
        # 1. Directional Accuracy
        directional_df = df[df['predicted_direction'].notna()].copy()
        if not directional_df.empty:
            directional_df['actual_direction'] = directional_df['actual_return'].apply(
                lambda x: 'up' if x > 0 else ('down' if x < 0 else 'neutral')
            )
            
            correct_direction = (directional_df['predicted_direction'] == directional_df['actual_direction']).sum()
            total_directional = len(directional_df)
            directional_accuracy = correct_direction / total_directional if total_directional > 0 else 0.0
            
            # Calculate confidence interval (Wilson score interval)
            ci_lower, ci_upper = self._calculate_proportion_ci(correct_direction, total_directional)
            
            results[AccuracyMetric.DIRECTIONAL_ACCURACY] = AccuracyResults(
                metric_type=AccuracyMetric.DIRECTIONAL_ACCURACY,
                value=directional_accuracy,
                sample_size=total_directional,
                confidence_interval=(ci_lower, ci_upper),
                time_window=window_name,
                by_symbol=directional_df.groupby('symbol').apply(
                    lambda x: (x['predicted_direction'] == x['actual_direction']).mean()
                ).to_dict(),
                by_model=directional_df.groupby('model_id').apply(
                    lambda x: (x['predicted_direction'] == x['actual_direction']).mean()
                ).to_dict()
            )
        
        # 2. Hit Rate (within confidence intervals)
        interval_df = df[(df['confidence_lower'].notna()) & (df['confidence_upper'].notna())].copy()
        if not interval_df.empty:
            hits = (
                (interval_df['actual_price'] >= interval_df['confidence_lower']) &
                (interval_df['actual_price'] <= interval_df['confidence_upper'])
            ).sum()
            total_intervals = len(interval_df)
            hit_rate = hits / total_intervals if total_intervals > 0 else 0.0
            
            ci_lower, ci_upper = self._calculate_proportion_ci(hits, total_intervals)
            
            results[AccuracyMetric.HIT_RATE] = AccuracyResults(
                metric_type=AccuracyMetric.HIT_RATE,
                value=hit_rate,
                sample_size=total_intervals,
                confidence_interval=(ci_lower, ci_upper),
                time_window=window_name,
                by_symbol=interval_df.groupby('symbol').apply(
                    lambda x: ((x['actual_price'] >= x['confidence_lower']) & 
                              (x['actual_price'] <= x['confidence_upper'])).mean()
                ).to_dict()
            )
        
        # 3. RMSE (Root Mean Square Error)
        price_df = df[df['predicted_price'].notna()].copy()
        if not price_df.empty:
            squared_errors = (price_df['predicted_price'] - price_df['actual_price']) ** 2
            rmse = np.sqrt(squared_errors.mean())
            
            results[AccuracyMetric.RMSE] = AccuracyResults(
                metric_type=AccuracyMetric.RMSE,
                value=rmse,
                sample_size=len(price_df),
                confidence_interval=(0.0, 0.0),  # TODO: Bootstrap CI for RMSE
                time_window=window_name,
                by_symbol=price_df.groupby('symbol').apply(
                    lambda x: np.sqrt(((x['predicted_price'] - x['actual_price']) ** 2).mean())
                ).to_dict()
            )
        
        # 4. MAE (Mean Absolute Error)
        if not price_df.empty:
            absolute_errors = np.abs(price_df['predicted_price'] - price_df['actual_price'])
            mae = absolute_errors.mean()
            
            results[AccuracyMetric.MAE] = AccuracyResults(
                metric_type=AccuracyMetric.MAE,
                value=mae,
                sample_size=len(price_df),
                confidence_interval=(0.0, 0.0),
                time_window=window_name,
                by_symbol=price_df.groupby('symbol').apply(
                    lambda x: np.abs(x['predicted_price'] - x['actual_price']).mean()
                ).to_dict()
            )
        
        # 5. Win Rate (profitable predictions)
        return_df = df[df['predicted_return'].notna()].copy()
        if not return_df.empty:
            # Consider a win if prediction and actual have same sign and actual return > 0
            wins = (
                (return_df['predicted_return'] * return_df['actual_return'] > 0) &
                (return_df['actual_return'] > 0)
            ).sum()
            total_trades = len(return_df)
            win_rate = wins / total_trades if total_trades > 0 else 0.0
            
            ci_lower, ci_upper = self._calculate_proportion_ci(wins, total_trades)
            
            results[AccuracyMetric.WIN_RATE] = AccuracyResults(
                metric_type=AccuracyMetric.WIN_RATE,
                value=win_rate,
                sample_size=total_trades,
                confidence_interval=(ci_lower, ci_upper),
                time_window=window_name
            )
        
        # Store accuracy snapshots
        self._store_accuracy_snapshots(results)
        
        return results
    
    def _calculate_proportion_ci(self, successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for proportions"""
        if total == 0:
            return (0.0, 0.0)
        
        z = 1.96  # 95% confidence
        p = successes / total
        
        denominator = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denominator
        half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        
        return (max(0, centre - half_width), min(1, centre + half_width))
    
    def _store_accuracy_snapshots(self, results: Dict[AccuracyMetric, AccuracyResults]):
        """Store accuracy snapshots for historical tracking"""
        
        with sqlite3.connect(self.db_path) as conn:
            for metric, result in results.items():
                snapshot_id = str(uuid.uuid4())
                metadata = {
                    'by_symbol': result.by_symbol,
                    'by_model': result.by_model,
                    'by_horizon': result.by_horizon
                }
                
                conn.execute("""
                    INSERT INTO accuracy_snapshots (
                        snapshot_id, timestamp, window_period, metric_type,
                        value, sample_size, confidence_lower, confidence_upper, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot_id,
                    datetime.now().isoformat(),
                    result.time_window,
                    metric.value,
                    result.value,
                    result.sample_size,
                    result.confidence_interval[0],
                    result.confidence_interval[1],
                    json.dumps(metadata)
                ))
    
    def generate_accuracy_report(self, 
                                 window_days: int = 30,
                                 symbols: Optional[List[str]] = None) -> str:
        """Generate comprehensive accuracy report"""
        
        results = self.calculate_accuracy_metrics(window_days, symbols)
        
        if not results:
            return f"No accuracy data available for {window_days}-day window"
        
        report = []
        report.append("=" * 80)
        report.append("DCF SUITE v0201 - MODEL ACCURACY REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"📊 Analysis Period: Last {window_days} days")
        report.append(f"🗓️ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if symbols:
            report.append(f"🎯 Symbols: {', '.join(symbols)}")
        else:
            report.append("🎯 Symbols: All")
        
        report.append("")
        report.append("📈 ACCURACY METRICS")
        report.append("-" * 50)
        
        # Main metrics
        for metric, result in results.items():
            if metric == AccuracyMetric.DIRECTIONAL_ACCURACY:
                report.append(f"Directional Accuracy:     {result.value:.1%} ± {(result.confidence_interval[1] - result.confidence_interval[0])/2:.1%}")
                report.append(f"  Sample size: {result.sample_size} predictions")
                
            elif metric == AccuracyMetric.HIT_RATE:
                report.append(f"Range Hit Rate:           {result.value:.1%} ± {(result.confidence_interval[1] - result.confidence_interval[0])/2:.1%}")
                report.append(f"  Sample size: {result.sample_size} predictions")
                
            elif metric == AccuracyMetric.RMSE:
                report.append(f"RMSE:                     ${result.value:.2f}")
                
            elif metric == AccuracyMetric.MAE:
                report.append(f"MAE:                      ${result.value:.2f}")
                
            elif metric == AccuracyMetric.WIN_RATE:
                report.append(f"Win Rate:                 {result.value:.1%} ± {(result.confidence_interval[1] - result.confidence_interval[0])/2:.1%}")
        
        # Symbol breakdown
        if AccuracyMetric.DIRECTIONAL_ACCURACY in results and results[AccuracyMetric.DIRECTIONAL_ACCURACY].by_symbol:
            report.append("")
            report.append("📊 BY SYMBOL BREAKDOWN")
            report.append("-" * 30)
            
            symbol_data = results[AccuracyMetric.DIRECTIONAL_ACCURACY].by_symbol
            for symbol, accuracy in sorted(symbol_data.items(), key=lambda x: x[1], reverse=True):
                report.append(f"{symbol:>6}: {accuracy:.1%} directional accuracy")
        
        # Model breakdown
        if AccuracyMetric.DIRECTIONAL_ACCURACY in results and results[AccuracyMetric.DIRECTIONAL_ACCURACY].by_model:
            report.append("")
            report.append("🤖 BY MODEL BREAKDOWN")
            report.append("-" * 25)
            
            model_data = results[AccuracyMetric.DIRECTIONAL_ACCURACY].by_model
            for model, accuracy in sorted(model_data.items(), key=lambda x: x[1], reverse=True):
                report.append(f"{model:>15}: {accuracy:.1%} directional accuracy")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_rolling_accuracy(self, metric: AccuracyMetric, days: int = 30) -> pd.DataFrame:
        """Get rolling accuracy over time"""
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM accuracy_snapshots 
                WHERE metric_type = ? AND window_period = ?
                ORDER BY timestamp ASC
            """, conn, params=(metric.value, f"{days}d"))
        
        if df.empty:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def integration_with_governance(self) -> Dict[str, Any]:
        """Integration point for governance framework"""
        
        # Calculate key metrics for governance
        recent_metrics = self.calculate_accuracy_metrics(window_days=7)  # Last week
        monthly_metrics = self.calculate_accuracy_metrics(window_days=30)  # Last month
        
        governance_data = {
            'last_updated': datetime.now().isoformat(),
            'model_performance': {
                'recent_directional_accuracy': recent_metrics.get(AccuracyMetric.DIRECTIONAL_ACCURACY, AccuracyResults(AccuracyMetric.DIRECTIONAL_ACCURACY, 0.0, 0, (0.0, 0.0), '7d')).value,
                'recent_hit_rate': recent_metrics.get(AccuracyMetric.HIT_RATE, AccuracyResults(AccuracyMetric.HIT_RATE, 0.0, 0, (0.0, 0.0), '7d')).value,
                'monthly_directional_accuracy': monthly_metrics.get(AccuracyMetric.DIRECTIONAL_ACCURACY, AccuracyResults(AccuracyMetric.DIRECTIONAL_ACCURACY, 0.0, 0, (0.0, 0.0), '30d')).value,
                'monthly_hit_rate': monthly_metrics.get(AccuracyMetric.HIT_RATE, AccuracyResults(AccuracyMetric.HIT_RATE, 0.0, 0, (0.0, 0.0), '30d')).value,
            },
            'alerts': []
        }
        
        # Generate alerts based on performance
        recent_dir_acc = governance_data['model_performance']['recent_directional_accuracy']
        monthly_dir_acc = governance_data['model_performance']['monthly_directional_accuracy']
        
        if recent_dir_acc < 0.5 and recent_metrics.get(AccuracyMetric.DIRECTIONAL_ACCURACY):
            if recent_metrics[AccuracyMetric.DIRECTIONAL_ACCURACY].sample_size >= 10:
                governance_data['alerts'].append({
                    'type': 'performance_degradation',
                    'severity': 'warning',
                    'message': f'Recent directional accuracy below 50%: {recent_dir_acc:.1%}',
                    'recommendation': 'Consider model retraining or feature investigation'
                })
        
        if monthly_dir_acc < 0.55 and monthly_metrics.get(AccuracyMetric.DIRECTIONAL_ACCURACY):
            if monthly_metrics[AccuracyMetric.DIRECTIONAL_ACCURACY].sample_size >= 30:
                governance_data['alerts'].append({
                    'type': 'model_underperformance',
                    'severity': 'error',
                    'message': f'Monthly directional accuracy concerning: {monthly_dir_acc:.1%}',
                    'recommendation': 'Immediate model review and retraining required'
                })
        
        return governance_data


# Integration with existing systems
def create_enhanced_prediction_with_tracking(ticker: str, 
                                           model_id: str = "enhanced_tf_v2.1",
                                           horizon_days: int = 5,
                                           tracker: Optional[LiveAccuracyTracker] = None) -> Tuple[Dict[str, Any], str]:
    """
    Enhanced prediction function that integrates accuracy tracking
    Uses ALL available features and modules from the DCF Suite
    """
    
    if tracker is None:
        tracker = LiveAccuracyTracker()
    
    try:
        # Import all available modules
        from dcf_lab.ai_price_forecast import AIPriceForecast
        
        # Create comprehensive prediction using all features
        ai_forecaster = AIPriceForecast()
        prediction_result = ai_forecaster.predict_price_with_confidence(
            ticker=ticker,
            target_date=horizon_days,
            include_macro=True,
            include_subsidiary=True,
            include_options=True
        )
        
        # Create prediction record for tracking
        prediction_record = PredictionRecord(
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            symbol=ticker,
            model_id=model_id,
            model_version="v2.1",
            prediction_type=PredictionType.MULTI_HORIZON,
            horizon_days=horizon_days,
            predicted_price=prediction_result.get('predicted_price'),
            predicted_direction=prediction_result.get('predicted_direction'),
            confidence_lower=prediction_result.get('confidence_interval', [None, None])[0],
            confidence_upper=prediction_result.get('confidence_interval', [None, None])[1],
            confidence_level=0.95,
            current_price=prediction_result.get('current_price', 0.0),
            predicted_return=prediction_result.get('predicted_return'),
            volatility_estimate=prediction_result.get('volatility_estimate'),
            features_hash=prediction_result.get('features_hash'),
            feature_count=prediction_result.get('feature_count', 0)
        )
        
        # Log prediction for tracking
        prediction_id = tracker.log_prediction(prediction_record)
        
        # Enhance result with tracking info
        enhanced_result = {
            **prediction_result,
            'tracking': {
                'prediction_id': prediction_id,
                'tracking_enabled': True,
                'model_id': model_id,
                'horizon_days': horizon_days
            }
        }
        
        return enhanced_result, prediction_id
        
    except Exception as e:
        logger.error(f"Error creating enhanced prediction: {e}")
        
        # Fallback prediction
        fallback_result = {
            'symbol': ticker,
            'predicted_price': 150.0,  # Mock
            'predicted_direction': 'up',
            'confidence_interval': [145.0, 155.0],
            'current_price': 148.0,
            'predicted_return': 0.0135,
            'error': str(e)
        }
        
        return fallback_result, "fallback"


if __name__ == "__main__":
    # Example usage
    tracker = LiveAccuracyTracker()
    
    # Generate sample predictions for testing
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    for symbol in symbols:
        result, pred_id = create_enhanced_prediction_with_tracking(
            ticker=symbol,
            horizon_days=5,
            tracker=tracker
        )
        print(f"Created prediction {pred_id} for {symbol}")
    
    # Generate accuracy report
    report = tracker.generate_accuracy_report(window_days=30)
    print(report)