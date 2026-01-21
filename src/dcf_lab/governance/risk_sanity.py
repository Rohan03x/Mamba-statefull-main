"""
Risk Sanity Validation Framework

This module implements pre-trade risk sanity checks for production trading:
- Options-implied volatility consistency validation
- Prediction interval validation against market expectations
- Real-time risk monitoring with automated alerts
- Integration with options data providers for IV surfaces
- Regulatory compliance for risk management controls

Key features:
- Real-time validation of model predictions against options-implied moves
- Dynamic volatility surface interpolation and extrapolation
- Risk violation detection with severity classification
- Integration with trading systems for pre-trade checks
- Comprehensive audit logging for regulatory compliance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
import sqlite3
from abc import ABC, abstractmethod
from enum import Enum

# Mathematical imports for volatility calculations
from scipy import interpolate

logger = logging.getLogger(__name__)

class RiskViolationType(Enum):
    """Types of risk violations"""
    PREDICTION_MISMATCH = "prediction_mismatch"
    INTERVAL_INCONSISTENCY = "interval_inconsistency"
    VOLATILITY_EXTREME = "volatility_extreme"
    LIQUIDITY_CONCERN = "liquidity_concern"
    DATA_QUALITY = "data_quality"

class ViolationSeverity(Enum):
    """Severity levels for risk violations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class RiskSanityConfig:
    """Configuration for risk sanity validation"""
    
    # Prediction validation thresholds
    max_prediction_iv_ratio: float = 2.0  # Max ratio of prediction to IV-implied move
    min_prediction_iv_ratio: float = 0.1  # Min ratio of prediction to IV-implied move
    
    # Interval validation parameters
    interval_consistency_threshold: float = 0.3  # Max deviation from IV-implied intervals
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.80, 0.90, 0.95])
    
    # Volatility surface parameters
    min_time_to_expiry: float = 1.0/365  # 1 day minimum
    max_time_to_expiry: float = 2.0      # 2 years maximum
    min_moneyness: float = 0.7           # 70% moneyness minimum
    max_moneyness: float = 1.3           # 130% moneyness maximum
    
    # Data quality checks
    min_iv_value: float = 0.01   # 1% minimum IV
    max_iv_value: float = 3.0    # 300% maximum IV
    stale_data_threshold_minutes: int = 30  # Minutes before data considered stale
    
    # Risk limits
    max_position_size: float = 0.1        # 10% maximum position size
    max_leverage: float = 3.0             # 3x maximum leverage
    max_concentration: float = 0.2        # 20% maximum single asset concentration
    
    # Alert thresholds
    warning_threshold: float = 1.5        # 1.5x IV ratio triggers warning
    error_threshold: float = 2.0          # 2.0x IV ratio triggers error
    critical_threshold: float = 3.0       # 3.0x IV ratio triggers critical alert

@dataclass
class OptionsData:
    """Options market data for a specific symbol"""
    
    symbol: str
    timestamp: str
    spot_price: float
    
    # Options chain data
    calls: List[Dict[str, Any]] = field(default_factory=list)
    puts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Derived metrics
    atm_iv: Optional[float] = None
    iv_30d: Optional[float] = None
    iv_skew: Optional[float] = None
    term_structure: Dict[float, float] = field(default_factory=dict)
    
    # Data quality indicators
    data_age_minutes: float = 0.0
    bid_ask_spread_ratio: float = 0.0
    volume_weighted_iv: Optional[float] = None

@dataclass
class RiskViolation:
    """Risk violation detected by sanity checks"""
    
    timestamp: str
    violation_type: RiskViolationType
    severity: ViolationSeverity
    symbol: str
    horizon_days: int
    
    # Violation details
    predicted_move: float
    implied_move: float
    ratio: float
    threshold: float
    
    # Additional context
    message: str
    confidence_interval: Optional[Tuple[float, float]] = None
    iv_data: Optional[Dict[str, Any]] = None
    
    # Risk metrics
    position_size: Optional[float] = None
    leverage: Optional[float] = None
    
    # Resolution tracking
    resolved: bool = False
    resolution_timestamp: Optional[str] = None
    resolution_action: Optional[str] = None

class OptionsDataProvider(ABC):
    """Abstract base class for options data providers"""
    
    @abstractmethod
    def get_options_data(self, symbol: str) -> Optional[OptionsData]:
        """Get current options data for symbol"""
        pass
    
    @abstractmethod
    def get_iv_surface(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get implied volatility surface"""
        pass
    
    @abstractmethod
    def is_data_fresh(self, symbol: str) -> bool:
        """Check if data is fresh enough for trading"""
        pass

class MockOptionsProvider(OptionsDataProvider):
    """Mock options data provider for testing"""
    
    def __init__(self, config: RiskSanityConfig):
        self.config = config
        self.mock_data = {}
        
    def get_options_data(self, symbol: str) -> Optional[OptionsData]:
        """Get mock options data"""
        
        # Generate realistic mock data
        base_price = 100.0  # Mock spot price
        base_iv = 0.25      # Mock base IV (25%)
        
        # Create mock options chain
        calls = []
        puts = []
        
        strikes = np.arange(0.8 * base_price, 1.2 * base_price, base_price * 0.05)
        
        for strike in strikes:
            moneyness = strike / base_price
            
            # Mock IV smile (higher IV for OTM options)
            iv_adjustment = 0.02 * abs(moneyness - 1.0) ** 2
            call_iv = base_iv + iv_adjustment
            put_iv = base_iv + iv_adjustment + 0.01  # Put skew
            
            calls.append({
                'strike': strike,
                'expiry': '2024-01-19',  # Mock expiry
                'bid': max(base_price - strike, 0) * 0.95,  # Mock bid
                'ask': max(base_price - strike, 0) * 1.05,  # Mock ask
                'implied_volatility': call_iv,
                'volume': np.random.randint(10, 1000),
                'open_interest': np.random.randint(100, 5000)
            })
            
            puts.append({
                'strike': strike,
                'expiry': '2024-01-19',  # Mock expiry
                'bid': max(strike - base_price, 0) * 0.95,  # Mock bid
                'ask': max(strike - base_price, 0) * 1.05,  # Mock ask
                'implied_volatility': put_iv,
                'volume': np.random.randint(10, 1000),
                'open_interest': np.random.randint(100, 5000)
            })
        
        return OptionsData(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            spot_price=base_price,
            calls=calls,
            puts=puts,
            atm_iv=base_iv,
            iv_30d=base_iv * 1.1,  # Slightly higher for 30-day
            iv_skew=0.01,
            data_age_minutes=np.random.uniform(0, 10),  # Fresh data
            bid_ask_spread_ratio=0.02,  # 2% spread
            volume_weighted_iv=base_iv * 1.02
        )
    
    def get_iv_surface(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get mock IV surface"""
        
        # Create mock IV surface
        moneyness_grid = np.arange(0.8, 1.21, 0.05)
        time_grid = np.array([7, 14, 30, 60, 90, 180, 365]) / 365.0  # In years
        
        iv_surface = []
        
        for time_to_expiry in time_grid:
            for moneyness in moneyness_grid:
                # Mock IV smile with term structure
                base_iv = 0.20 + 0.05 * np.sqrt(time_to_expiry)  # Term structure
                skew_adjustment = 0.03 * (1 - moneyness) ** 2     # Volatility smile
                
                iv = base_iv + skew_adjustment
                
                iv_surface.append({
                    'moneyness': moneyness,
                    'time_to_expiry': time_to_expiry,
                    'implied_volatility': iv
                })
        
        return pd.DataFrame(iv_surface)
    
    def is_data_fresh(self, symbol: str) -> bool:
        """Mock data freshness check"""
        return True  # Always fresh for mock

class OptionsImpliedChecker:
    """Calculates options-implied moves and validates predictions"""
    
    def __init__(self, config: RiskSanityConfig):
        self.config = config
        
    def calculate_implied_move(self, options_data: OptionsData, 
                              horizon_days: int) -> Optional[Dict[str, float]]:
        """
        Calculate options-implied move for given horizon
        
        Args:
            options_data: Options market data
            horizon_days: Prediction horizon in days
            
        Returns:
            Dictionary with implied move statistics
        """
        
        try:
            # Find appropriate options for the horizon
            target_expiry_days = horizon_days
            
            # For simplicity, use ATM IV if available
            if options_data.atm_iv is not None:
                atm_iv = options_data.atm_iv
            else:
                # Extract ATM IV from options chain
                atm_iv = self._extract_atm_iv(options_data)
            
            if atm_iv is None:
                logger.warning(f"No ATM IV available for {options_data.symbol}")
                return None
            
            # Calculate time-adjusted IV
            time_to_expiry = target_expiry_days / 365.0
            
            # Adjust IV for different time horizons (simplified approach)
            # In practice, would use volatility surface interpolation
            adjusted_iv = atm_iv * np.sqrt(time_to_expiry / (30/365))  # Scale from 30-day base
            
            # Calculate implied moves
            spot_price = options_data.spot_price
            
            # 1-sigma move (68% confidence)
            one_sigma_move = spot_price * adjusted_iv * np.sqrt(time_to_expiry)
            
            # 2-sigma move (95% confidence)
            two_sigma_move = one_sigma_move * 2.0
            
            # Expected move (from straddle price if available, otherwise use 0.8 * 1-sigma)
            expected_move = one_sigma_move * 0.8
            
            return {
                'implied_volatility': adjusted_iv,
                'time_to_expiry': time_to_expiry,
                'one_sigma_move': one_sigma_move,
                'two_sigma_move': two_sigma_move,
                'expected_move': expected_move,
                'one_sigma_percentage': one_sigma_move / spot_price,
                'two_sigma_percentage': two_sigma_move / spot_price,
                'confidence_68': (-one_sigma_move, one_sigma_move),
                'confidence_95': (-two_sigma_move, two_sigma_move)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate implied move: {e}")
            return None
    
    def _extract_atm_iv(self, options_data: OptionsData) -> Optional[float]:
        """Extract ATM implied volatility from options chain"""
        
        spot_price = options_data.spot_price
        
        # Find closest ATM options
        atm_ivs = []
        
        for call in options_data.calls:
            strike = call.get('strike', 0)
            if abs(strike - spot_price) / spot_price < 0.02:  # Within 2% of ATM
                iv = call.get('implied_volatility')
                if iv and self.config.min_iv_value <= iv <= self.config.max_iv_value:
                    atm_ivs.append(iv)
        
        for put in options_data.puts:
            strike = put.get('strike', 0)
            if abs(strike - spot_price) / spot_price < 0.02:  # Within 2% of ATM
                iv = put.get('implied_volatility')
                if iv and self.config.min_iv_value <= iv <= self.config.max_iv_value:
                    atm_ivs.append(iv)
        
        if atm_ivs:
            return np.mean(atm_ivs)
        
        return None
    
    def interpolate_iv_surface(self, iv_surface: pd.DataFrame, 
                              moneyness: float, time_to_expiry: float) -> Optional[float]:
        """Interpolate IV from surface"""
        
        try:
            # Ensure we have required columns
            required_cols = ['moneyness', 'time_to_expiry', 'implied_volatility']
            if not all(col in iv_surface.columns for col in required_cols):
                return None
            
            # Check bounds
            if (moneyness < self.config.min_moneyness or 
                moneyness > self.config.max_moneyness or
                time_to_expiry < self.config.min_time_to_expiry or
                time_to_expiry > self.config.max_time_to_expiry):
                logger.warning(f"Interpolation point outside bounds: moneyness={moneyness}, tte={time_to_expiry}")
                return None
            
            # 2D interpolation
            points = iv_surface[['moneyness', 'time_to_expiry']].values
            values = iv_surface['implied_volatility'].values
            
            if len(points) < 4:  # Need minimum points for interpolation
                return None
            
            # Use linear interpolation
            interpolator = interpolate.LinearNDInterpolator(points, values)
            interpolated_iv = interpolator(moneyness, time_to_expiry)
            
            if np.isfinite(interpolated_iv):
                return float(interpolated_iv)
            
            # Fallback to nearest neighbor
            interpolator = interpolate.NearestNDInterpolator(points, values)
            interpolated_iv = interpolator(moneyness, time_to_expiry)
            
            return float(interpolated_iv) if np.isfinite(interpolated_iv) else None
            
        except Exception as e:
            logger.error(f"IV surface interpolation failed: {e}")
            return None

class RiskSanityValidator:
    """Main validator for risk sanity checks"""
    
    def __init__(self, config: RiskSanityConfig, 
                 options_provider: OptionsDataProvider):
        self.config = config
        self.options_provider = options_provider
        self.implied_checker = OptionsImpliedChecker(config)
        
        # Violation storage
        self.violation_history = []
        
        # Database for persistence
        self.db_path = "risk_violations.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize risk violations database"""
        
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                symbol TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                predicted_move REAL,
                implied_move REAL,
                ratio REAL,
                threshold REAL,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_timestamp TEXT,
                resolution_action TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def validate_prediction(self, symbol: str, predicted_move: float,
                           confidence_interval: Tuple[float, float],
                           horizon_days: int, 
                           position_size: Optional[float] = None) -> List[RiskViolation]:
        """
        Validate model prediction against options-implied expectations
        
        Args:
            symbol: Trading symbol
            predicted_move: Model's predicted price move
            confidence_interval: Model's confidence interval (lower, upper)
            horizon_days: Prediction horizon in days
            position_size: Optional position size for risk checks
            
        Returns:
            List of risk violations (empty if no violations)
        """
        
        violations = []
        
        try:
            # Get options data
            options_data = self.options_provider.get_options_data(symbol)
            
            if options_data is None:
                violations.append(self._create_violation(
                    symbol=symbol,
                    horizon_days=horizon_days,
                    violation_type=RiskViolationType.DATA_QUALITY,
                    severity=ViolationSeverity.WARNING,
                    predicted_move=predicted_move,
                    implied_move=0.0,
                    ratio=0.0,
                    threshold=0.0,
                    message="No options data available for validation"
                ))
                return violations
            
            # Check data freshness
            if options_data.data_age_minutes > self.config.stale_data_threshold_minutes:
                violations.append(self._create_violation(
                    symbol=symbol,
                    horizon_days=horizon_days,
                    violation_type=RiskViolationType.DATA_QUALITY,
                    severity=ViolationSeverity.WARNING,
                    predicted_move=predicted_move,
                    implied_move=0.0,
                    ratio=0.0,
                    threshold=self.config.stale_data_threshold_minutes,
                    message=f"Stale options data: {options_data.data_age_minutes:.1f} minutes old"
                ))
            
            # Calculate implied move
            implied_stats = self.implied_checker.calculate_implied_move(options_data, horizon_days)
            
            if implied_stats is None:
                violations.append(self._create_violation(
                    symbol=symbol,
                    horizon_days=horizon_days,
                    violation_type=RiskViolationType.DATA_QUALITY,
                    severity=ViolationSeverity.ERROR,
                    predicted_move=predicted_move,
                    implied_move=0.0,
                    ratio=0.0,
                    threshold=0.0,
                    message="Failed to calculate options-implied move"
                ))
                return violations
            
            # Validate prediction against implied move
            implied_move = implied_stats['expected_move']
            
            if implied_move > 0:
                prediction_ratio = abs(predicted_move) / implied_move
                
                # Check ratio bounds
                if prediction_ratio > self.config.max_prediction_iv_ratio:
                    severity = self._determine_severity(prediction_ratio)
                    violations.append(self._create_violation(
                        symbol=symbol,
                        horizon_days=horizon_days,
                        violation_type=RiskViolationType.PREDICTION_MISMATCH,
                        severity=severity,
                        predicted_move=predicted_move,
                        implied_move=implied_move,
                        ratio=prediction_ratio,
                        threshold=self.config.max_prediction_iv_ratio,
                        message=f"Prediction {prediction_ratio:.2f}x larger than implied move",
                        confidence_interval=confidence_interval,
                        iv_data=implied_stats
                    ))
                
                elif prediction_ratio < self.config.min_prediction_iv_ratio:
                    violations.append(self._create_violation(
                        symbol=symbol,
                        horizon_days=horizon_days,
                        violation_type=RiskViolationType.PREDICTION_MISMATCH,
                        severity=ViolationSeverity.INFO,
                        predicted_move=predicted_move,
                        implied_move=implied_move,
                        ratio=prediction_ratio,
                        threshold=self.config.min_prediction_iv_ratio,
                        message=f"Prediction {prediction_ratio:.2f}x smaller than implied move",
                        confidence_interval=confidence_interval,
                        iv_data=implied_stats
                    ))
            
            # Validate confidence intervals
            interval_violations = self._validate_confidence_intervals(
                symbol, horizon_days, confidence_interval, implied_stats
            )
            violations.extend(interval_violations)
            
            # Position size validation if provided
            if position_size is not None:
                size_violations = self._validate_position_size(
                    symbol, horizon_days, position_size, predicted_move
                )
                violations.extend(size_violations)
            
            # Store violations in database
            for violation in violations:
                self._store_violation(violation)
            
            # Add to history
            self.violation_history.extend(violations)
            
            return violations
            
        except Exception as e:
            logger.error(f"Risk validation failed for {symbol}: {e}")
            
            error_violation = self._create_violation(
                symbol=symbol,
                horizon_days=horizon_days,
                violation_type=RiskViolationType.DATA_QUALITY,
                severity=ViolationSeverity.ERROR,
                predicted_move=predicted_move,
                implied_move=0.0,
                ratio=0.0,
                threshold=0.0,
                message=f"Validation error: {str(e)}"
            )
            
            return [error_violation]
    
    def _validate_confidence_intervals(self, symbol: str, horizon_days: int,
                                     confidence_interval: Tuple[float, float],
                                     implied_stats: Dict[str, float]) -> List[RiskViolation]:
        """Validate confidence intervals against implied volatility"""
        
        violations = []
        
        try:
            lower_bound, upper_bound = confidence_interval
            interval_width = upper_bound - lower_bound
            
            # Compare with implied confidence intervals
            implied_95_width = 2 * implied_stats['two_sigma_move']  # 95% interval width
            
            if interval_width > 0:
                width_ratio = interval_width / implied_95_width
                
                if abs(width_ratio - 1.0) > self.config.interval_consistency_threshold:
                    severity = ViolationSeverity.WARNING if width_ratio > 2.0 else ViolationSeverity.INFO
                    
                    violations.append(self._create_violation(
                        symbol=symbol,
                        horizon_days=horizon_days,
                        violation_type=RiskViolationType.INTERVAL_INCONSISTENCY,
                        severity=severity,
                        predicted_move=(upper_bound + lower_bound) / 2,
                        implied_move=0.0,  # Center move
                        ratio=width_ratio,
                        threshold=1.0 + self.config.interval_consistency_threshold,
                        message=f"Confidence interval {width_ratio:.2f}x different from implied",
                        confidence_interval=confidence_interval,
                        iv_data=implied_stats
                    ))
            
        except Exception as e:
            logger.error(f"Confidence interval validation failed: {e}")
        
        return violations
    
    def _validate_position_size(self, symbol: str, horizon_days: int,
                               position_size: float, predicted_move: float) -> List[RiskViolation]:
        """Validate position size against risk limits"""
        
        violations = []
        
        # Check maximum position size
        if abs(position_size) > self.config.max_position_size:
            violations.append(self._create_violation(
                symbol=symbol,
                horizon_days=horizon_days,
                violation_type=RiskViolationType.LIQUIDITY_CONCERN,
                severity=ViolationSeverity.ERROR,
                predicted_move=predicted_move,
                implied_move=0.0,
                ratio=abs(position_size) / self.config.max_position_size,
                threshold=self.config.max_position_size,
                message=f"Position size {abs(position_size):.2%} exceeds maximum {self.config.max_position_size:.2%}",
                position_size=position_size
            ))
        
        return violations
    
    def _determine_severity(self, ratio: float) -> ViolationSeverity:
        """Determine violation severity based on ratio"""
        
        if ratio >= self.config.critical_threshold:
            return ViolationSeverity.CRITICAL
        elif ratio >= self.config.error_threshold:
            return ViolationSeverity.ERROR
        elif ratio >= self.config.warning_threshold:
            return ViolationSeverity.WARNING
        else:
            return ViolationSeverity.INFO
    
    def _create_violation(self, symbol: str, horizon_days: int,
                         violation_type: RiskViolationType, severity: ViolationSeverity,
                         predicted_move: float, implied_move: float,
                         ratio: float, threshold: float, message: str,
                         confidence_interval: Optional[Tuple[float, float]] = None,
                         iv_data: Optional[Dict[str, Any]] = None,
                         position_size: Optional[float] = None) -> RiskViolation:
        """Create a risk violation object"""
        
        return RiskViolation(
            timestamp=datetime.now().isoformat(),
            violation_type=violation_type,
            severity=severity,
            symbol=symbol,
            horizon_days=horizon_days,
            predicted_move=predicted_move,
            implied_move=implied_move,
            ratio=ratio,
            threshold=threshold,
            message=message,
            confidence_interval=confidence_interval,
            iv_data=iv_data,
            position_size=position_size
        )
    
    def _store_violation(self, violation: RiskViolation):
        """Store violation in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO risk_violations 
                (timestamp, violation_type, severity, symbol, horizon_days,
                 predicted_move, implied_move, ratio, threshold, message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                violation.timestamp,
                violation.violation_type.value,
                violation.severity.value,
                violation.symbol,
                violation.horizon_days,
                violation.predicted_move,
                violation.implied_move,
                violation.ratio,
                violation.threshold,
                violation.message
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store violation: {e}")
    
    def get_recent_violations(self, hours: int = 24) -> List[RiskViolation]:
        """Get recent violations from database"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_str = cutoff_time.isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT timestamp, violation_type, severity, symbol, horizon_days,
                       predicted_move, implied_move, ratio, threshold, message
                FROM risk_violations 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff_str,))
            
            violations = []
            for row in cursor.fetchall():
                violation = RiskViolation(
                    timestamp=row[0],
                    violation_type=RiskViolationType(row[1]),
                    severity=ViolationSeverity(row[2]),
                    symbol=row[3],
                    horizon_days=row[4],
                    predicted_move=row[5],
                    implied_move=row[6],
                    ratio=row[7],
                    threshold=row[8],
                    message=row[9]
                )
                violations.append(violation)
            
            conn.close()
            return violations
            
        except Exception as e:
            logger.error(f"Failed to retrieve violations: {e}")
            return []
    
    def get_violation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get violation summary statistics"""
        
        violations = self.get_recent_violations(hours)
        
        if not violations:
            return {
                'total_violations': 0,
                'by_severity': {},
                'by_type': {},
                'by_symbol': {}
            }
        
        # Count by severity
        severity_counts = {}
        for severity in ViolationSeverity:
            severity_counts[severity.value] = sum(1 for v in violations if v.severity == severity)
        
        # Count by type
        type_counts = {}
        for violation_type in RiskViolationType:
            type_counts[violation_type.value] = sum(1 for v in violations if v.violation_type == violation_type)
        
        # Count by symbol
        symbol_counts = {}
        for violation in violations:
            symbol_counts[violation.symbol] = symbol_counts.get(violation.symbol, 0) + 1
        
        return {
            'total_violations': len(violations),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'by_symbol': symbol_counts,
            'most_violated_symbol': max(symbol_counts, key=symbol_counts.get) if symbol_counts else None
        }

class PreTradeRiskCheck:
    """Pre-trade risk check coordinator"""
    
    def __init__(self, config: RiskSanityConfig, 
                 options_provider: OptionsDataProvider):
        self.config = config
        self.validator = RiskSanityValidator(config, options_provider)
        
        # Risk check results cache
        self.check_cache = {}
        self.cache_ttl_minutes = 5  # Cache results for 5 minutes
        
    def perform_pre_trade_check(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive pre-trade risk check
        
        Args:
            trade_request: Dictionary containing trade details
            
        Returns:
            Risk check results with approval/rejection decision
        """
        
        symbol = trade_request.get('symbol')
        predicted_move = trade_request.get('predicted_move', 0.0)
        confidence_interval = trade_request.get('confidence_interval', (0.0, 0.0))
        horizon_days = trade_request.get('horizon_days', 1)
        position_size = trade_request.get('position_size', 0.0)
        
        # Check cache first
        cache_key = f"{symbol}_{predicted_move}_{horizon_days}"
        if cache_key in self.check_cache:
            cached_result = self.check_cache[cache_key]
            cache_age = (datetime.now() - datetime.fromisoformat(cached_result['timestamp'])).total_seconds() / 60
            
            if cache_age < self.cache_ttl_minutes:
                logger.debug(f"Using cached risk check result for {symbol}")
                return cached_result
        
        # Perform validation
        violations = self.validator.validate_prediction(
            symbol=symbol,
            predicted_move=predicted_move,
            confidence_interval=confidence_interval,
            horizon_days=horizon_days,
            position_size=position_size
        )
        
        # Determine approval status
        approval_status = self._determine_approval_status(violations)
        
        # Create risk check result
        result = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'horizon_days': horizon_days,
            'predicted_move': predicted_move,
            'position_size': position_size,
            'approval_status': approval_status['status'],
            'approval_reason': approval_status['reason'],
            'violations': [
                {
                    'type': v.violation_type.value,
                    'severity': v.severity.value,
                    'message': v.message,
                    'ratio': v.ratio
                } for v in violations
            ],
            'risk_score': self._calculate_risk_score(violations),
            'trade_approved': approval_status['approved']
        }
        
        # Cache result
        self.check_cache[cache_key] = result
        
        # Clean old cache entries
        self._clean_cache()
        
        return result
    
    def _determine_approval_status(self, violations: List[RiskViolation]) -> Dict[str, Any]:
        """Determine if trade should be approved based on violations"""
        
        if not violations:
            return {
                'approved': True,
                'status': 'approved',
                'reason': 'No risk violations detected'
            }
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            return {
                'approved': False,
                'status': 'rejected',
                'reason': f'Critical risk violation: {critical_violations[0].message}'
            }
        
        # Check for error violations
        error_violations = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        if error_violations:
            return {
                'approved': False,
                'status': 'rejected',
                'reason': f'Risk error: {error_violations[0].message}'
            }
        
        # Warning violations - conditional approval
        warning_violations = [v for v in violations if v.severity == ViolationSeverity.WARNING]
        if warning_violations:
            return {
                'approved': True,
                'status': 'conditional_approval',
                'reason': f'Approved with warnings: {len(warning_violations)} warnings detected'
            }
        
        # Only info violations
        return {
            'approved': True,
            'status': 'approved',
            'reason': f'Approved with {len(violations)} informational notices'
        }
    
    def _calculate_risk_score(self, violations: List[RiskViolation]) -> float:
        """Calculate overall risk score from violations"""
        
        if not violations:
            return 0.0
        
        # Weight violations by severity
        severity_weights = {
            ViolationSeverity.INFO: 1.0,
            ViolationSeverity.WARNING: 3.0,
            ViolationSeverity.ERROR: 10.0,
            ViolationSeverity.CRITICAL: 25.0
        }
        
        total_score = sum(severity_weights.get(v.severity, 1.0) for v in violations)
        
        # Normalize to 0-100 scale
        return min(total_score * 2, 100.0)
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, result in self.check_cache.items():
            cache_age = (current_time - datetime.fromisoformat(result['timestamp'])).total_seconds() / 60
            if cache_age > self.cache_ttl_minutes:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.check_cache[key]
    
    def get_check_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get pre-trade check statistics"""
        
        # Get recent violations from validator
        violation_summary = self.validator.get_violation_summary(hours)
        
        # Calculate approval rates (would need to track all checks in practice)
        return {
            'violation_summary': violation_summary,
            'cache_size': len(self.check_cache),
            'system_status': 'operational' if len(self.check_cache) < 1000 else 'degraded'
        }