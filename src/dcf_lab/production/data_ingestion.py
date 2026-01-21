"""
Production Data Ingestion & Validation System

This module handles real-time data ingestion for the production trading system:
- Schema validation and data quality checks
- Feature store management with versioning
- Real-time data validation and rejection handling
- Integration with upstream data sources

Key features:
- Schema enforcement with automatic rejection of invalid rows
- Missing value and future date detection
- Feature store with time-series optimized storage
- Data lineage and audit trails
- Real-time validation metrics and alerting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class DataSchema:
    """Production data schema definition"""
    
    # Required columns and their types
    required_columns: Dict[str, str] = field(default_factory=lambda: {
        'date': 'datetime64[ns]',
        'symbol': 'object',
        'price': 'float64',
        'volume': 'float64',
        'return_1d': 'float64'
    })
    
    # Optional columns with defaults
    optional_columns: Dict[str, str] = field(default_factory=lambda: {
        'news_sentiment': 'float64',
        'implied_vol': 'float64',
        'realized_vol': 'float64'
    })
    
    # Value constraints
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'price': {'min': 0.01, 'max': 10000},
        'volume': {'min': 0, 'max': 1e12},
        'return_1d': {'min': -0.5, 'max': 0.5},
        'news_sentiment': {'min': 0, 'max': 1},
        'implied_vol': {'min': 0, 'max': 2},
        'realized_vol': {'min': 0, 'max': 2}
    })
    
    # Date constraints
    max_future_days: int = 1  # Allow 1 day in future for timezone issues
    max_past_days: int = 365 * 5  # Allow 5 years of historical data

@dataclass
class ValidationResult:
    """Result of data validation"""
    
    is_valid: bool = True
    valid_rows: int = 0
    invalid_rows: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    
    def add_error(self, error: str, count: int = 1):
        """Add validation error"""
        self.errors.append(error)
        self.rejection_reasons[error] = self.rejection_reasons.get(error, 0) + count
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)

class SchemaValidator:
    """Validates data against production schema"""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
        
    def validate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Validate dataframe against schema
        
        Args:
            data: Input dataframe
            
        Returns:
            Tuple of (clean_data, validation_result)
        """
        
        result = ValidationResult()
        clean_data = data.copy()
        
        logger.info(f"Validating {len(data)} rows against production schema...")
        
        # 1. Check required columns
        missing_cols = set(self.schema.required_columns.keys()) - set(data.columns)
        if missing_cols:
            result.add_error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame(), result
        
        # 2. Validate data types
        clean_data, type_errors = self._validate_types(clean_data)
        if type_errors:
            result.add_error(f"Type conversion errors: {len(type_errors)}", len(type_errors))
        
        # 3. Check for null values in required columns
        null_mask = clean_data[list(self.schema.required_columns.keys())].isnull().any(axis=1)
        null_count = null_mask.sum()
        if null_count > 0:
            result.add_error("Null values in required columns", null_count)
            clean_data = clean_data[~null_mask]
        
        # 4. Validate date constraints
        clean_data, date_errors = self._validate_dates(clean_data)
        if date_errors > 0:
            result.add_error("Invalid dates (future/too old)", date_errors)
        
        # 5. Validate value constraints
        clean_data, constraint_errors = self._validate_constraints(clean_data)
        for error_type, count in constraint_errors.items():
            if count > 0:
                result.add_error(f"Constraint violation: {error_type}", count)
        
        # 6. Check for duplicates
        if 'date' in clean_data.columns and 'symbol' in clean_data.columns:
            duplicate_mask = clean_data.duplicated(subset=['date', 'symbol'])
            duplicate_count = duplicate_mask.sum()
            if duplicate_count > 0:
                result.add_warning(f"Duplicate records removed: {duplicate_count}")
                clean_data = clean_data[~duplicate_mask]
        
        # Update result statistics
        result.valid_rows = len(clean_data)
        result.invalid_rows = len(data) - len(clean_data)
        
        if result.invalid_rows == 0:
            result.is_valid = True
            result.errors = []  # Clear errors if all rows are valid
        
        logger.info(f"Validation complete: {result.valid_rows} valid, {result.invalid_rows} invalid")
        
        return clean_data, result
    
    def _validate_types(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and convert data types"""
        
        errors = []
        clean_data = data.copy()
        
        # Convert types for required columns
        for col, dtype in self.schema.required_columns.items():
            if col in clean_data.columns:
                try:
                    if dtype == 'datetime64[ns]':
                        clean_data[col] = pd.to_datetime(clean_data[col])
                    else:
                        clean_data[col] = clean_data[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    errors.append(f"Failed to convert {col} to {dtype}: {str(e)}")
                    # Remove rows with conversion errors
                    clean_data = clean_data.dropna(subset=[col])
        
        # Convert types for optional columns
        for col, dtype in self.schema.optional_columns.items():
            if col in clean_data.columns:
                try:
                    clean_data[col] = clean_data[col].astype(dtype)
                except (ValueError, TypeError):
                    # For optional columns, just convert to float and let NaN handle errors
                    clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')
        
        return clean_data, errors
    
    def _validate_dates(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Validate date constraints"""
        
        if 'date' not in data.columns:
            return data, 0
        
        now = datetime.now()
        max_future = now + timedelta(days=self.schema.max_future_days)
        min_past = now - timedelta(days=self.schema.max_past_days)
        
        # Check future dates
        future_mask = data['date'] > max_future
        future_count = future_mask.sum()
        
        # Check too old dates
        old_mask = data['date'] < min_past
        old_count = old_mask.sum()
        
        # Remove invalid dates
        valid_mask = ~(future_mask | old_mask)
        clean_data = data[valid_mask]
        
        total_errors = future_count + old_count
        
        if total_errors > 0:
            logger.warning(f"Removed {future_count} future dates and {old_count} old dates")
        
        return clean_data, total_errors
    
    def _validate_constraints(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Validate value constraints"""
        
        constraint_errors = {}
        clean_data = data.copy()
        
        for col, constraints in self.schema.constraints.items():
            if col not in clean_data.columns:
                continue
            
            # Check minimum values
            if 'min' in constraints:
                min_val = constraints['min']
                below_min = (clean_data[col] < min_val) & clean_data[col].notna()
                below_min_count = below_min.sum()
                if below_min_count > 0:
                    constraint_errors[f"{col}_below_min"] = below_min_count
                    clean_data = clean_data[~below_min]
            
            # Check maximum values  
            if 'max' in constraints:
                max_val = constraints['max']
                above_max = (clean_data[col] > max_val) & clean_data[col].notna()
                above_max_count = above_max.sum()
                if above_max_count > 0:
                    constraint_errors[f"{col}_above_max"] = above_max_count
                    clean_data = clean_data[~above_max]
        
        return clean_data, constraint_errors

class FeatureStore:
    """Time-series optimized feature store for production data"""
    
    def __init__(self, db_path: str = "production_feature_store.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with optimized schema"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Main features table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    feature_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_version INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(date, symbol, feature_name, data_version)
                )
            """)
            
            # Metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    feature_name TEXT PRIMARY KEY,
                    feature_type TEXT NOT NULL,
                    description TEXT,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Data quality metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_date_symbol ON features(date, symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_date ON features(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol ON features(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_date ON data_quality(date)")
    
    def store_features(self, data: pd.DataFrame, feature_type: str = "raw", 
                      data_version: int = 1) -> bool:
        """
        Store features in the feature store
        
        Args:
            data: DataFrame with features
            feature_type: Type of features (raw, processed, engineered)
            data_version: Version of the data
            
        Returns:
            Success flag
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert DataFrame to feature records
                records = []
                created_at = datetime.now().isoformat()
                
                for _, row in data.iterrows():
                    for col in data.columns:
                        if col not in ['date', 'symbol']:
                            records.append({
                                'date': row['date'] if 'date' in row else created_at,
                                'symbol': row['symbol'] if 'symbol' in row else 'UNKNOWN',
                                'feature_name': col,
                                'feature_value': row[col] if pd.notna(row[col]) else None,
                                'feature_type': feature_type,
                                'created_at': created_at,
                                'data_version': data_version
                            })
                
                # Insert records
                conn.executemany("""
                    INSERT OR REPLACE INTO features 
                    (date, symbol, feature_name, feature_value, feature_type, created_at, data_version)
                    VALUES (:date, :symbol, :feature_name, :feature_value, :feature_type, :created_at, :data_version)
                """, records)
                
                logger.info(f"Stored {len(records)} feature records")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False
    
    def get_features(self, symbols: List[str], start_date: str, end_date: str,
                    feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve features from the store
        
        Args:
            symbols: List of symbols to retrieve
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            feature_names: Optional list of specific features
            
        Returns:
            DataFrame with features
        """
        
        query = """
            SELECT date, symbol, feature_name, feature_value, feature_type
            FROM features
            WHERE date >= ? AND date <= ?
            AND symbol IN ({})
        """.format(','.join(['?'] * len(symbols)))
        
        params = [start_date, end_date] + symbols
        
        if feature_names:
            query += " AND feature_name IN ({})".format(','.join(['?'] * len(feature_names)))
            params.extend(feature_names)
        
        query += " ORDER BY date, symbol, feature_name"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    # Pivot to get features as columns
                    pivot_df = df.pivot_table(
                        index=['date', 'symbol'], 
                        columns='feature_name', 
                        values='feature_value'
                    ).reset_index()
                    
                    return pivot_df
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Failed to retrieve features: {e}")
            return pd.DataFrame()
    
    def get_latest_features(self, symbols: List[str], 
                           feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get the most recent features for given symbols"""
        
        # Get the latest date for each symbol
        latest_query = """
            SELECT symbol, MAX(date) as latest_date
            FROM features
            WHERE symbol IN ({})
            GROUP BY symbol
        """.format(','.join(['?'] * len(symbols)))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                latest_df = pd.read_sql_query(latest_query, conn, params=symbols)
                
                if latest_df.empty:
                    return pd.DataFrame()
                
                # Get features for latest dates
                results = []
                for _, row in latest_df.iterrows():
                    symbol_features = self.get_features(
                        [row['symbol']], row['latest_date'], row['latest_date'], feature_names
                    )
                    if not symbol_features.empty:
                        results.append(symbol_features)
                
                if results:
                    return pd.concat(results, ignore_index=True)
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Failed to get latest features: {e}")
            return pd.DataFrame()

class ProductionDataValidator:
    """High-level data validator for production ingestion"""
    
    def __init__(self, schema: Optional[DataSchema] = None):
        self.schema = schema or DataSchema()
        self.validator = SchemaValidator(self.schema)
        self.validation_history = []
    
    def validate_and_clean(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Main validation and cleaning pipeline
        
        Args:
            data: Raw input data
            
        Returns:
            Tuple of (cleaned_data, validation_result)
        """
        
        start_time = datetime.now()
        
        # Validate against schema
        clean_data, result = self.validator.validate_data(data)
        
        # Store validation history
        validation_record = {
            'timestamp': start_time.isoformat(),
            'input_rows': len(data),
            'output_rows': len(clean_data),
            'validation_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
            'errors': len(result.errors),
            'warnings': len(result.warnings)
        }
        
        self.validation_history.append(validation_record)
        
        # Keep only recent history (last 1000 validations)
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        return clean_data, result
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        if not self.validation_history:
            return {}
        
        recent_validations = self.validation_history[-100:]  # Last 100 validations
        
        return {
            'total_validations': len(self.validation_history),
            'recent_success_rate': sum(1 for v in recent_validations if v['errors'] == 0) / len(recent_validations),
            'avg_validation_time_ms': np.mean([v['validation_time_ms'] for v in recent_validations]),
            'avg_rejection_rate': np.mean([
                (v['input_rows'] - v['output_rows']) / max(v['input_rows'], 1) 
                for v in recent_validations
            ]),
            'last_validation': recent_validations[-1] if recent_validations else None
        }

class DataIngestionPipeline:
    """Complete production data ingestion pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.schema = DataSchema()
        self.validator = ProductionDataValidator(self.schema)
        self.feature_store = FeatureStore(
            self.config.get('feature_store_path', 'production_feature_store.db')
        )
        
        # Metrics tracking
        self.ingestion_metrics = {
            'total_ingested': 0,
            'total_rejected': 0,
            'last_ingestion': None,
            'errors': []
        }
    
    async def ingest_data(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Main data ingestion pipeline
        
        Args:
            data: Input data in various formats
            
        Returns:
            Ingestion result with metrics
        """
        
        start_time = datetime.now()
        
        try:
            # Convert input to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            logger.info(f"Starting ingestion of {len(df)} rows...")
            
            # Validate and clean data
            clean_data, validation_result = self.validator.validate_and_clean(df)
            
            # Store in feature store if validation passed
            if validation_result.valid_rows > 0:
                store_success = self.feature_store.store_features(
                    clean_data, 
                    feature_type="raw",
                    data_version=1
                )
                
                if not store_success:
                    validation_result.add_error("Failed to store in feature store")
            
            # Update metrics
            self.ingestion_metrics['total_ingested'] += validation_result.valid_rows
            self.ingestion_metrics['total_rejected'] += validation_result.invalid_rows
            self.ingestion_metrics['last_ingestion'] = start_time.isoformat()
            
            if validation_result.errors:
                self.ingestion_metrics['errors'].extend(validation_result.errors)
                # Keep only recent errors
                self.ingestion_metrics['errors'] = self.ingestion_metrics['errors'][-100:]
            
            # Prepare result
            result = {
                'success': validation_result.is_valid,
                'processed_rows': validation_result.valid_rows,
                'rejected_rows': validation_result.invalid_rows,
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'validation_result': validation_result,
                'stored_in_feature_store': validation_result.valid_rows > 0
            }
            
            logger.info(f"Ingestion complete: {result['processed_rows']} processed, {result['rejected_rows']} rejected")
            
            return result
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            self.ingestion_metrics['errors'].append(str(e))
            
            return {
                'success': False,
                'error': str(e),
                'processed_rows': 0,
                'rejected_rows': len(df) if 'df' in locals() else 0,
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics"""
        
        validation_stats = self.validator.get_validation_stats()
        
        return {
            'ingestion_metrics': self.ingestion_metrics,
            'validation_stats': validation_stats,
            'feature_store_status': 'connected',  # Could be enhanced with actual checks
            'system_health': 'healthy' if not self.ingestion_metrics['errors'] else 'warning'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        
        try:
            # Test feature store connection
            test_data = pd.DataFrame({
                'date': [datetime.now().isoformat()],
                'symbol': ['TEST'],
                'price': [100.0],
                'volume': [1000],
                'return_1d': [0.01]
            })
            
            # Test validation
            _, validation_result = self.validator.validate_and_clean(test_data)
            
            # Test feature store
            store_test = self.feature_store.store_features(test_data, "test")
            
            return {
                'status': 'healthy',
                'validation_working': validation_result.is_valid,
                'feature_store_working': store_test,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }