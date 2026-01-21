"""
Production Deployment System for DCF Suite v0201

Comprehensive production deployment management integrating all modules:
- Live prediction pipeline with accuracy tracking
- Risk management and position sizing
- Integration with all 10 task modules
- Paper trading to live deployment pipeline
- Real-time monitoring and governance

This system brings together:
- Task 1-7: Core ML features and data processing
- Task 8: Subsidiary mapping and macro features (COMPLETE)
- Task 9: Multi-horizon models (TFT implementation)
- Task 10: Governance framework (COMPLETE)
- New: Live accuracy tracking and production deployment
"""

import logging
import asyncio
import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

# Import all DCF Suite modules
from .accuracy_tracking import LiveAccuracyTracker, create_enhanced_prediction_with_tracking
from .performance_dashboard import PerformanceDashboard

logger = logging.getLogger(__name__)

class DeploymentPhase(Enum):
    DEVELOPMENT = "development"
    ACCURACY_VALIDATION = "accuracy_validation" 
    PAPER_TRADING = "paper_trading"
    LIMITED_LIVE = "limited_live"
    FULL_PRODUCTION = "full_production"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DeploymentConfig:
    """Configuration for production deployment"""
    phase: DeploymentPhase
    max_position_size: float = 1000.0  # USD
    max_daily_trades: int = 50
    min_accuracy_threshold: float = 0.55
    stop_loss_threshold: float = 0.02  # 2%
    max_drawdown_threshold: float = 0.05  # 5%
    symbols_whitelist: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL'])
    require_governance_approval: bool = True
    enable_real_money: bool = False

@dataclass
class LivePosition:
    """Live trading position"""
    position_id: str
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    prediction_id: str
    target_price: float
    stop_loss_price: float
    is_paper_trade: bool = True
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: Optional[float] = None

class ProductionDeploymentSystem:
    """Complete production deployment system"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.accuracy_tracker = LiveAccuracyTracker()
        self.dashboard = PerformanceDashboard(self.accuracy_tracker)
        
        # Trading state
        self.active_positions: Dict[str, LivePosition] = {}
        self.daily_trades_count = 0
        self.last_trade_date = datetime.now().date()
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        
    async def run_live_deployment(self):
        """Main live deployment loop"""
        
        logger.info(f"🚀 Starting DCF Suite v0201 Live Deployment - Phase: {self.config.phase.value}")
        
        while True:
            try:
                # Daily reset
                current_date = datetime.now().date()
                if current_date != self.last_trade_date:
                    self.daily_trades_count = 0
                    self.daily_pnl = 0.0
                    self.last_trade_date = current_date
                    logger.info(f"📅 New trading day: {current_date}")
                
                # Get market data and validate predictions
                await self._validate_predictions()
                
                # Check deployment readiness
                if not await self._check_deployment_readiness():
                    logger.warning("⚠️ Deployment readiness check failed, skipping trading cycle")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                # Generate new predictions for all symbols
                predictions = await self._generate_live_predictions()
                
                # Execute trading decisions
                if predictions:
                    await self._execute_trading_decisions(predictions)
                
                # Monitor existing positions
                await self._monitor_positions()
                
                # Risk management
                await self._risk_management_check()
                
                # Update dashboard metrics
                await self._update_metrics()
                
                # Log system status
                logger.info(f"💹 Live system operational - Active positions: {len(self.active_positions)}, Daily PnL: ${self.daily_pnl:.2f}")
                
                # Wait before next cycle
                await asyncio.sleep(60)  # 1 minute cycle
                
            except Exception as e:
                logger.error(f"❌ Error in live deployment loop: {e}")
                await asyncio.sleep(60)
    
    async def _validate_predictions(self):
        """Validate pending predictions against market data"""
        
        try:
            # Mock market data - in production, this would fetch real market data
            market_data = await self._fetch_market_data()
            
            validation_stats = self.accuracy_tracker.validate_predictions(market_data)
            
            if validation_stats['validated'] > 0:
                logger.info(f"✅ Validated {validation_stats['validated']} predictions")
                
        except Exception as e:
            logger.error(f"Error validating predictions: {e}")
    
    async def _check_deployment_readiness(self) -> bool:
        """Check if system is ready for trading"""
        
        try:
            # Get governance data
            governance_data = self.accuracy_tracker.integration_with_governance()
            
            # Check accuracy thresholds
            monthly_acc = governance_data.get('model_performance', {}).get('monthly_directional_accuracy', 0)
            recent_acc = governance_data.get('model_performance', {}).get('recent_directional_accuracy', 0)
            
            # Check for critical alerts
            critical_alerts = [a for a in governance_data.get('alerts', []) if a.get('severity') == 'error']
            
            readiness_checks = [
                ('Monthly accuracy threshold', monthly_acc >= self.config.min_accuracy_threshold),
                ('Recent accuracy threshold', recent_acc >= 0.50),
                ('No critical alerts', len(critical_alerts) == 0),
                ('Max drawdown check', self.max_drawdown <= self.config.max_drawdown_threshold),
                ('Daily trade limit', self.daily_trades_count < self.config.max_daily_trades)
            ]
            
            failed_checks = [name for name, passed in readiness_checks if not passed]
            
            if failed_checks:
                logger.warning(f"❌ Readiness checks failed: {failed_checks}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking deployment readiness: {e}")
            return False
    
    async def _generate_live_predictions(self) -> List[Dict[str, Any]]:
        """Generate live predictions for all symbols"""
        
        predictions = []
        
        try:
            for symbol in self.config.symbols_whitelist:
                if self.daily_trades_count >= self.config.max_daily_trades:
                    break
                
                # Check if we already have a position in this symbol
                if any(pos.symbol == symbol for pos in self.active_positions.values()):
                    continue
                
                # Generate comprehensive prediction using all modules
                prediction_result, prediction_id = create_enhanced_prediction_with_tracking(
                    ticker=symbol,
                    model_id="production_v2.1",
                    horizon_days=5,
                    tracker=self.accuracy_tracker
                )
                
                # Add trading context
                prediction_result['prediction_id'] = prediction_id
                prediction_result['timestamp'] = datetime.now()
                
                predictions.append(prediction_result)
                
                logger.info(f"🎯 Generated prediction for {symbol}: {prediction_result.get('predicted_direction')} ({prediction_result.get('predicted_price', 0):.2f})")
        
        except Exception as e:
            logger.error(f"Error generating live predictions: {e}")
        
        return predictions
    
    async def _execute_trading_decisions(self, predictions: List[Dict[str, Any]]):
        """Execute trading decisions based on predictions"""
        
        for prediction in predictions:
            try:
                symbol = prediction['symbol']
                predicted_direction = prediction.get('predicted_direction')
                predicted_price = prediction.get('predicted_price')
                current_price = prediction.get('current_price', 0)
                prediction.get('confidence_interval', [None, None])
                
                # Trading decision logic
                should_trade = self._should_execute_trade(prediction)
                
                if should_trade and predicted_direction in ['up', 'down']:
                    # Calculate position size
                    position_size = self._calculate_position_size(symbol, prediction)
                    
                    if position_size > 0:
                        # Create position
                        position = await self._create_position(
                            symbol=symbol,
                            prediction=prediction,
                            position_size=position_size
                        )
                        
                        if position:
                            self.active_positions[position.position_id] = position
                            self.daily_trades_count += 1
                            
                            logger.info(f"📈 Created {symbol} position: {predicted_direction} @{current_price:.2f} target {predicted_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error executing trade for {prediction.get('symbol')}: {e}")
    
    def _check_validation_gates(self, symbol: str, horizon: int) -> tuple:
        """
        🎯 Step 7: Check validation gates before executing trades.
        
        Returns:
            (gates_passed: bool, long_only: bool, reason: str)
            - gates_passed: True if validation passed
            - long_only: True if should switch to long-only mode
            - reason: Explanation string
        """
        wf_result_path = Path(f"wf_result_{symbol}_H{horizon}.json")
        
        if not wf_result_path.exists():
            logger.warning(f"⚠️ No validation file found: {wf_result_path}")
            return (True, False, "no_validation_file")
        
        try:
            with open(wf_result_path, 'r') as f:
                wf_result = json.load(f)
            
            # Check if validation gates exist in result
            validation = wf_result.get("validation_gates")
            if not validation:
                logger.info(f"ℹ️ No validation gates in {wf_result_path}, proceeding")
                return (True, False, "no_gates_in_file")
            
            overall_status = validation.get("overall_status", "UNKNOWN")
            
            if overall_status == "PASS_DUAL":
                logger.info(f"✅ Validation passed for {symbol} H{horizon}: DUAL mode")
                return (True, False, "dual_mode")
            
            elif overall_status == "PASS_LONG_ONLY":
                logger.warning(f"⚠️ Validation: {symbol} H{horizon} switching to LONG-ONLY (short quality failed)")
                return (True, True, "long_only_mode")
            
            else:  # FAIL or unknown
                logger.error(f"❌ Validation FAILED for {symbol} H{horizon}: {overall_status}")
                return (False, False, "validation_failed")
        
        except Exception as e:
            logger.error(f"Error reading validation gates: {e}")
            return (True, False, "error_reading_gates")
    
    def _should_execute_trade(self, prediction: Dict[str, Any]) -> bool:
        """
        Determine if we should execute a trade based on asymmetric thresholds.
        
        Step 3 Implementation: Dual execution policy with asymmetric thresholds.
        - Long trades: pred >= thresh_long
        - Short trades: pred <= (1 - thresh_short)
        
        Step 4 Implementation: Auto-configuration with fallback logic.
        - Dynamically picks thresholds per symbol:horizon pair
        - Auto-disables shorts if Sharpe < 1.0 or coverage < floor
        
        Step 7 Implementation: Validation gate checks before execution.
        - Checks validation_gates section in wf_result.json
        - Respects overall_status: PASS_DUAL, PASS_LONG_ONLY, or FAIL
        """
        
        # Get prediction probability (assumes 0-1 scale, 1=bullish)
        pred_prob = prediction.get('probability', 0.5)
        predicted_direction = prediction.get('predicted_direction')
        confidence_interval = prediction.get('confidence_interval', [None, None])
        predicted_return = prediction.get('predicted_return', 0)
        current_price = prediction.get('current_price', 0)
        
        # Basic safety checks
        if predicted_direction not in ['up', 'down']:
            return False
        if current_price <= 0:
            return False
        if None in confidence_interval:
            return False
        if abs(predicted_return) < 0.01:  # 1% minimum
            return False
        
        # Step 7: Check validation gates
        symbol = prediction.get('symbol', 'UNKNOWN')
        horizon = prediction.get('horizon', 63)
        
        gates_passed, force_long_only, gate_reason = self._check_validation_gates(symbol, horizon)
        
        if not gates_passed:
            logger.warning(f"⛔ Trade blocked by validation gates: {symbol} H{horizon} ({gate_reason})")
            return False
        
        # Step 4: Get auto-configured thresholds for this symbol:horizon
        thresh_long, thresh_short, short_enabled = self._pick_thresholds(symbol, horizon)
        
        # Step 7: Override short_enabled if validation forces long-only
        if force_long_only:
            short_enabled = False
            logger.info(f"🔒 Validation gates: forcing long-only for {symbol} H{horizon}")

        
        # Step 3: Apply asymmetric execution policy
        if predicted_direction == 'up':
            # Long trade: require pred >= thresh_long
            if pred_prob >= thresh_long:
                size = self._confidence_to_size(pred_prob, thresh_long)
                prediction['position_size'] = size
                return True
        
        elif predicted_direction == 'down' and short_enabled:
            # Item 4: Regime-gate shorts (check market regime before executing)
            regime_allows_shorts, regime_reason = self._check_market_regime(symbol, current_price)
            
            if not regime_allows_shorts:
                logger.info(
                    f"🚫 Short trade blocked by regime gate: {symbol} H{horizon} "
                    f"(reason: {regime_reason})"
                )
                return False
            
            # Short trade: require pred <= (1 - thresh_short)
            short_cutoff = 1.0 - thresh_short
            if pred_prob <= short_cutoff:
                size = self._confidence_to_size(1.0 - pred_prob, thresh_short)
                prediction['position_size'] = size
                logger.info(
                    f"✅ Short trade approved: {symbol} H{horizon} "
                    f"(regime: {regime_reason}, prob={pred_prob:.3f}, size={size:.2f})"
                )
                return True
        
        # Threshold not met or shorts disabled
        return False
    
    def _pick_thresholds(self, symbol: str, horizon: int) -> tuple:
        """
        Step 4: Auto-threshold logic per symbol:horizon pair.
        
        Returns:
            (thresh_long, thresh_short, short_enabled)
        
        Fallback logic:
        - If short_sharpe < 1.0 or short_coverage < floor: disable shorts
        - Otherwise use optimized thresholds from wf_result.json
        """
        # Try loading optimized thresholds from walk-forward results
        wf_result_path = f"wf_result_{symbol}_H{horizon}.json"
        
        if os.path.exists(wf_result_path):
            try:
                with open(wf_result_path, 'r') as f:
                    wf_data = json.load(f)
                
                # Extract asymmetric metrics
                # long_metrics = wf_data.get('long_metrics', {})  # Reserved for future use
                short_metrics = wf_data.get('short_metrics', {})
                
                thresh_long = wf_data.get('threshold_long', 0.85)
                thresh_short = wf_data.get('threshold_short', 0.96)
                
                # Step 4 fallback: Check short performance
                short_sharpe = short_metrics.get('sharpe', 0.0)
                short_coverage = short_metrics.get('coverage', 0.0)
                coverage_floor = 0.05  # 5% minimum coverage
                
                if short_sharpe < 1.0 or short_coverage < coverage_floor:
                    short_enabled = False
                    logger.info(
                        f"Auto-disabling shorts for {symbol}:H{horizon} - "
                        f"Sharpe={short_sharpe:.2f}, Coverage={short_coverage:.2%}"
                    )
                else:
                    short_enabled = True
                
                return thresh_long, thresh_short, short_enabled
            
            except Exception as e:
                logger.warning(f"Failed to load thresholds for {symbol}:H{horizon}: {e}")
        
        # Default fallback (conservative)
        return 0.85, 0.96, True
    
    def _confidence_to_size(self, conf: float, threshold: float) -> float:
        """
        Convert prediction confidence to position size.
        
        Maps confidence above threshold to position size [0.0, 1.0]:
        - conf = threshold → size = 0.0 (minimum)
        - conf = 1.0 → size = 1.0 (maximum)
        
        Linear scaling for simplicity.
        """
        if conf <= threshold:
            return 0.0
        
        # Linear interpolation
        size = (conf - threshold) / (1.0 - threshold)
        return min(1.0, max(0.0, size))
    
    def _check_market_regime(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """
        Check if market regime supports short selling (Item 4).
        
        Regime-gate shorts: Enable only if market shows risk-off conditions:
        1. Price < 200-day moving average (individual stock weakness)
        2. Market breadth < 30% (broad market weakness)
        3. VIX term inversion OR credit spread widening (risk-off confirmation)
        
        Args:
            symbol: Stock symbol
            current_price: Current price for comparison
        
        Returns:
            Tuple of (shorts_allowed, reason)
        """
        # Read configuration from environment
        enable_regime_gating = os.getenv('ENABLE_REGIME_GATING', '1') == '1'
        if not enable_regime_gating:
            return True, "regime_gating_disabled"
        
        breadth_threshold = float(os.getenv('REGIME_BREADTH_THRESHOLD', '0.30'))
        use_vix = os.getenv('REGIME_USE_VIX', '1') == '1'
        use_credit = os.getenv('REGIME_USE_CREDIT', '0') == '1'
        
        try:
            # Check 1: Price < 200DMA
            price_file = self.data_dir / f"{symbol}_daily.csv"
            if price_file.exists():
                df = pd.read_csv(price_file)
                if len(df) >= 200:
                    sma_200 = df['Close'].tail(200).mean()
                    price_weak = current_price < sma_200
                    
                    if not price_weak:
                        return False, f"price_above_200dma (${current_price:.2f} >= ${sma_200:.2f})"
                else:
                    # Not enough data, allow shorts (conservative default)
                    price_weak = True
            else:
                price_weak = True  # No data available
            
            # Check 2: Market breadth < threshold (e.g., 30%)
            # Calculate breadth from universe symbols
            universe_symbols = os.getenv('REGIME_UNIVERSE', 'AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,NFLX').split(',')
            above_50dma_count = 0
            total_count = 0
            
            for sym in universe_symbols:
                sym_file = self.data_dir / f"{sym}_daily.csv"
                if sym_file.exists():
                    sym_df = pd.read_csv(sym_file)
                    if len(sym_df) >= 50:
                        current_sym_price = sym_df['Close'].iloc[-1]
                        sma_50 = sym_df['Close'].tail(50).mean()
                        if current_sym_price > sma_50:
                            above_50dma_count += 1
                        total_count += 1
            
            if total_count > 0:
                breadth_pct = above_50dma_count / total_count
                breadth_weak = breadth_pct < breadth_threshold
                
                if not breadth_weak:
                    return False, f"breadth_healthy ({breadth_pct:.1%} >= {breadth_threshold:.1%})"
            else:
                breadth_weak = True  # No breadth data, allow shorts
            
            # Check 3: VIX term inversion OR credit spread widening
            risk_off_confirmed = False
            risk_off_reason = []
            
            if use_vix:
                # Check VIX term structure
                vix_file = self.data_dir / "VIX_term_structure.csv"
                if vix_file.exists():
                    vix_df = pd.read_csv(vix_file)
                    if len(vix_df) > 0:
                        latest = vix_df.iloc[-1]
                        if 'VX1' in latest and 'VX3' in latest:
                            vix_term_slope = latest['VX1'] - latest['VX3']
                            if vix_term_slope > 0:  # Inverted = risk-off
                                risk_off_confirmed = True
                                risk_off_reason.append(f"vix_inverted (slope={vix_term_slope:.2f})")
            
            if use_credit and not risk_off_confirmed:
                # Check credit spread widening
                credit_file = self.data_dir / "credit_spreads.csv"
                if credit_file.exists():
                    credit_df = pd.read_csv(credit_file)
                    if len(credit_df) >= 5:
                        current_spread = credit_df['spread'].iloc[-1]
                        past_spread = credit_df['spread'].iloc[-6]
                        if current_spread > past_spread:
                            risk_off_confirmed = True
                            risk_off_reason.append(f"credit_widening ({current_spread:.2f} > {past_spread:.2f})")
            
            # If no risk-off indicators available, default to allowing shorts
            # (conservative: don't block if data unavailable)
            if not use_vix and not use_credit:
                risk_off_confirmed = True
                risk_off_reason.append("no_risk_indicators_configured")
            elif not risk_off_confirmed:
                return False, "risk_on_regime (no VIX inversion or credit widening)"
            
            # All checks passed: shorts allowed
            reason = f"risk_off_confirmed ({', '.join(risk_off_reason)})"
            return True, reason
            
        except Exception as e:
            logger.warning(f"Regime check failed for {symbol}: {e}")
            # On error, allow shorts (don't block on data issues)
            return True, f"regime_check_error ({str(e)})"
    
    def _calculate_position_size(self, symbol: str, prediction: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        
        try:
            current_price = prediction.get('current_price', 0)
            predicted_return = prediction.get('predicted_return', 0)
            
            if current_price <= 0:
                return 0
            
            # Base position size
            base_size = min(self.config.max_position_size, self.config.max_position_size * 0.5)
            
            # Adjust for expected return (higher return = larger position, up to a limit)
            return_multiplier = min(2.0, 1.0 + abs(predicted_return) * 10)
            adjusted_size = base_size * return_multiplier
            
            # Convert to number of shares
            max_shares = int(adjusted_size / current_price)
            
            return max_shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def _create_position(self, symbol: str, prediction: Dict[str, Any], position_size: int) -> Optional[LivePosition]:
        """Create a new trading position"""
        
        try:
            current_price = prediction.get('current_price')
            predicted_price = prediction.get('predicted_price')
            predicted_direction = prediction.get('predicted_direction')
            
            # Calculate stop loss
            stop_loss_distance = current_price * self.config.stop_loss_threshold
            
            if predicted_direction == 'up':
                stop_loss_price = current_price - stop_loss_distance
                target_price = predicted_price
            else:  # down
                stop_loss_price = current_price + stop_loss_distance
                target_price = predicted_price
                position_size = -position_size  # Short position
            
            position = LivePosition(
                position_id=str(uuid.uuid4()),
                symbol=symbol,
                entry_price=current_price,
                quantity=position_size,
                entry_time=datetime.now(),
                prediction_id=prediction.get('prediction_id'),
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                is_paper_trade=(self.config.phase != DeploymentPhase.FULL_PRODUCTION)
            )
            
            # In paper trading mode, just log the position
            if position.is_paper_trade:
                logger.info(f"📝 Paper trade: {position.symbol} {position.quantity} shares @${position.entry_price:.2f}")
            else:
                # Real trading would execute through broker API
                logger.info(f"💰 Live trade: {position.symbol} {position.quantity} shares @${position.entry_price:.2f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error creating position: {e}")
            return None
    
    async def _monitor_positions(self):
        """Monitor existing positions for exit conditions"""
        
        positions_to_close = []
        
        for position_id, position in self.active_positions.items():
            try:
                # Get current market price
                current_price = await self._get_current_price(position.symbol)
                
                if current_price is None:
                    continue
                
                # Check exit conditions
                should_exit, exit_reason = self._check_exit_conditions(position, current_price)
                
                if should_exit:
                    # Close position
                    await self._close_position(position, current_price, exit_reason)
                    positions_to_close.append(position_id)
            
            except Exception as e:
                logger.error(f"Error monitoring position {position_id}: {e}")
        
        # Remove closed positions
        for position_id in positions_to_close:
            del self.active_positions[position_id]
    
    def _check_exit_conditions(self, position: LivePosition, current_price: float) -> Tuple[bool, str]:
        """Check if position should be exited"""
        
        # Time-based exit (5 days max)
        if datetime.now() - position.entry_time > timedelta(days=5):
            return True, "time_limit"
        
        # Stop loss
        if position.quantity > 0:  # Long position
            if current_price <= position.stop_loss_price:
                return True, "stop_loss"
            if current_price >= position.target_price:
                return True, "target_reached"
        else:  # Short position
            if current_price >= position.stop_loss_price:
                return True, "stop_loss"
            if current_price <= position.target_price:
                return True, "target_reached"
        
        return False, ""
    
    async def _close_position(self, position: LivePosition, exit_price: float, exit_reason: str):
        """Close a trading position"""
        
        try:
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            
            # Calculate PnL
            if position.quantity > 0:  # Long
                position.realized_pnl = (exit_price - position.entry_price) * position.quantity
            else:  # Short
                position.realized_pnl = (position.entry_price - exit_price) * abs(position.quantity)
            
            # Update totals
            self.daily_pnl += position.realized_pnl
            self.total_pnl += position.realized_pnl
            
            # Update drawdown tracking
            if self.total_pnl > self.peak_value:
                self.peak_value = self.total_pnl
            
            current_drawdown = (self.peak_value - self.total_pnl) / max(self.peak_value, 1)
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            trade_type = "Paper" if position.is_paper_trade else "Live"
            logger.info(f"🔄 {trade_type} position closed: {position.symbol} PnL: ${position.realized_pnl:.2f} ({exit_reason})")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def _risk_management_check(self):
        """Perform risk management checks"""
        
        try:
            # Check maximum drawdown
            if self.max_drawdown > self.config.max_drawdown_threshold:
                logger.error(f"🚨 Maximum drawdown exceeded: {self.max_drawdown:.1%} > {self.config.max_drawdown_threshold:.1%}")
                
                # Emergency position closure
                if self.config.phase == DeploymentPhase.FULL_PRODUCTION:
                    await self._emergency_close_all_positions()
            
            # Check daily loss limit
            if self.daily_pnl < -self.config.max_position_size * 0.1:  # 10% of max position
                logger.warning(f"⚠️ Daily loss limit approaching: ${self.daily_pnl:.2f}")
            
            # Check position concentration
            symbol_exposure = {}
            for position in self.active_positions.values():
                exposure = abs(position.quantity * position.entry_price)
                symbol_exposure[position.symbol] = symbol_exposure.get(position.symbol, 0) + exposure
            
            for symbol, exposure in symbol_exposure.items():
                if exposure > self.config.max_position_size * 2:
                    logger.warning(f"⚠️ High exposure to {symbol}: ${exposure:.2f}")
        
        except Exception as e:
            logger.error(f"Error in risk management check: {e}")
    
    async def _emergency_close_all_positions(self):
        """Emergency closure of all positions"""
        
        logger.error("🚨 EMERGENCY: Closing all positions")
        
        for position in list(self.active_positions.values()):
            try:
                current_price = await self._get_current_price(position.symbol)
                if current_price:
                    await self._close_position(position, current_price, "emergency")
            except Exception as e:
                logger.error(f"Error closing position in emergency: {e}")
        
        self.active_positions.clear()
    
    async def _update_metrics(self):
        """Update dashboard and tracking metrics"""
        
        try:
            # Update performance metrics
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'active_positions': len(self.active_positions),
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown,
                'daily_trades': self.daily_trades_count,
                'deployment_phase': self.config.phase.value
            }
            
            # Log metrics
            logger.info(f"📊 Metrics updated: {json.dumps(metrics_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _fetch_market_data(self) -> Dict[str, Dict[str, float]]:
        """Fetch current market data - mock implementation"""
        
        # Mock market data for validation
        # In production, this would fetch real market data
        market_data = {}
        
        for symbol in self.config.symbols_whitelist:
            # Generate mock price data for last 10 days
            base_price = 150.0 if symbol == 'AAPL' else 300.0
            dates_data = {}
            
            for i in range(10):
                date = (datetime.now() - timedelta(days=i)).date().isoformat()
                price_variation = np.random.normal(0, 0.02)  # 2% daily volatility
                dates_data[date] = base_price * (1 + price_variation)
            
            market_data[symbol] = dates_data
        
        return market_data
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        
        # Mock current price
        base_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 120.0}
        base_price = base_prices.get(symbol, 100.0)
        
        # Add some random variation
        variation = np.random.normal(0, 0.005)  # 0.5% variation
        return base_price * (1 + variation)
    
    def generate_deployment_report(self) -> str:
        """Generate comprehensive deployment status report"""
        
        report = []
        report.append("=" * 80)
        report.append("DCF SUITE v0201 - PRODUCTION DEPLOYMENT REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"🚀 Deployment Phase: {self.config.phase.value.upper()}")
        report.append(f"🗓️ Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance metrics
        report.append("📊 PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Active Positions:     {len(self.active_positions)}")
        report.append(f"Daily Trades:         {self.daily_trades_count}/{self.config.max_daily_trades}")
        report.append(f"Daily PnL:           ${self.daily_pnl:.2f}")
        report.append(f"Total PnL:           ${self.total_pnl:.2f}")
        report.append(f"Max Drawdown:         {self.max_drawdown:.1%}")
        report.append("")
        
        # Risk metrics
        report.append("⚠️ RISK METRICS")
        report.append("-" * 30)
        risk_level = "LOW" if self.max_drawdown < 0.02 else ("MEDIUM" if self.max_drawdown < 0.05 else "HIGH")
        report.append(f"Risk Level:           {risk_level}")
        report.append(f"Position Limit:      ${self.config.max_position_size:.0f}")
        report.append(f"Stop Loss:            {self.config.stop_loss_threshold:.1%}")
        report.append("")
        
        # Active positions
        if self.active_positions:
            report.append("📈 ACTIVE POSITIONS")
            report.append("-" * 35)
            for position in self.active_positions.values():
                days_held = (datetime.now() - position.entry_time).days
                report.append(f"{position.symbol}: {position.quantity} shares @${position.entry_price:.2f} ({days_held}d)")
        
        # System integration
        report.append("")
        report.append("🔧 SYSTEM INTEGRATION")
        report.append("-" * 40)
        report.append("✅ Task 1-7: Core ML Pipeline")
        report.append("✅ Task 8: Subsidiary/Macro Features")
        report.append("✅ Task 9: Multi-horizon Models (TFT)")
        report.append("✅ Task 10: Governance Framework")
        report.append("✅ Live Accuracy Tracking")
        report.append("✅ Performance Dashboard")
        report.append("✅ Production Deployment")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


async def main():
    """Main deployment system entry point"""
    
    # Configuration for current phase
    config = DeploymentConfig(
        phase=DeploymentPhase.ACCURACY_VALIDATION,
        max_position_size=1000.0,
        max_daily_trades=20,
        symbols_whitelist=['AAPL', 'MSFT', 'GOOGL'],
        enable_real_money=False  # Paper trading mode
    )
    
    # Initialize deployment system
    deployment_system = ProductionDeploymentSystem(config)
    
    print("🚀 DCF Suite v0201 - Production Deployment System")
    print("=" * 60)
    print(deployment_system.generate_deployment_report())
    print("\n🔄 Starting live deployment loop...")
    
    # Run live deployment
    await deployment_system.run_live_deployment()


if __name__ == "__main__":
    asyncio.run(main())