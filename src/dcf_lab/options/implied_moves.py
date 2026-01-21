"""
Implied Moves and Volatility Computation

This module computes implied moves from options market data, particularly
focusing on near-ATM straddles to derive 1σ move expectations that serve
as neutral priors for the options-anchored learning framework.

Key components:
- ATM straddle identification and pricing
- Implied volatility surface construction
- 1σ move computation from straddle premiums
- Options data processing and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptionsChain:
    """Options chain data structure"""
    calls: pd.DataFrame
    puts: pd.DataFrame
    spot_price: float
    expiration: datetime
    risk_free_rate: float
    dividend_yield: float = 0.0
    timestamp: datetime = None

@dataclass
class ATMStraddle:
    """At-the-money straddle information"""
    strike: float
    call_price: float
    put_price: float
    total_premium: float
    implied_vol: float
    days_to_expiry: int
    delta_neutral_strike: float
    
    @property
    def straddle_cost(self) -> float:
        """Total cost of the straddle"""
        return self.call_price + self.put_price
    
    @property
    def breakeven_range(self) -> Tuple[float, float]:
        """Breakeven range for the straddle"""
        return (self.strike - self.total_premium, self.strike + self.total_premium)

@dataclass
class ImpliedMove:
    """Implied move calculation results"""
    one_sigma_move: float
    one_sigma_percent: float
    two_sigma_move: float
    two_sigma_percent: float
    straddle_move: float
    straddle_percent: float
    confidence_interval: Tuple[float, float]
    calculation_method: str
    metadata: Dict

class ImpliedVolatilityCalculator:
    """Black-Scholes implied volatility calculator"""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        """
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:  # put
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return max(0, price)
    
    def calculate_implied_vol(self, market_price: float, S: float, K: float, 
                            T: float, r: float, option_type: str = 'call') -> Optional[float]:
        """
        Calculate implied volatility using Brent's method
        
        Args:
            market_price: Observed market price
            S, K, T, r: Black-Scholes parameters
            option_type: 'call' or 'put'
        """
        if market_price <= 0 or T <= 0:
            return None
        
        def objective(sigma):
            try:
                bs_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
                return (bs_price - market_price)**2
            except:
                return 1e10
        
        try:
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            if result.success and result.fun < self.tolerance:
                return result.x
        except:
            pass
        
        return None
    
    def calculate_chain_ivs(self, options_chain: OptionsChain) -> OptionsChain:
        """Calculate implied volatilities for entire options chain"""
        
        # Calculate time to expiration
        if options_chain.timestamp:
            time_diff = options_chain.expiration - options_chain.timestamp
        else:
            time_diff = options_chain.expiration - datetime.now()
        
        T = time_diff.days / 365.25
        
        # Process calls
        calls_iv = []
        for _, option in options_chain.calls.iterrows():
            iv = self.calculate_implied_vol(
                option['last_price'], 
                options_chain.spot_price,
                option['strike'],
                T,
                options_chain.risk_free_rate,
                'call'
            )
            calls_iv.append(iv)
        
        # Process puts  
        puts_iv = []
        for _, option in options_chain.puts.iterrows():
            iv = self.calculate_implied_vol(
                option['last_price'],
                options_chain.spot_price, 
                option['strike'],
                T,
                options_chain.risk_free_rate,
                'put'
            )
            puts_iv.append(iv)
        
        # Add IV columns
        calls_with_iv = options_chain.calls.copy()
        puts_with_iv = options_chain.puts.copy()
        calls_with_iv['implied_vol'] = calls_iv
        puts_with_iv['implied_vol'] = puts_iv
        
        return OptionsChain(
            calls=calls_with_iv,
            puts=puts_with_iv,
            spot_price=options_chain.spot_price,
            expiration=options_chain.expiration,
            risk_free_rate=options_chain.risk_free_rate,
            dividend_yield=options_chain.dividend_yield,
            timestamp=options_chain.timestamp
        )

class ATMStraddleFinder:
    """Find and analyze at-the-money straddles"""
    
    def __init__(self, iv_calculator: ImpliedVolatilityCalculator):
        self.iv_calculator = iv_calculator
    
    def find_atm_straddle(self, options_chain: OptionsChain, 
                         method: str = 'closest_strike') -> Optional[ATMStraddle]:
        """
        Find the best ATM straddle representation
        
        Args:
            options_chain: Options chain data
            method: 'closest_strike', 'delta_neutral', or 'interpolated'
        """
        if method == 'closest_strike':
            return self._find_closest_strike_straddle(options_chain)
        elif method == 'delta_neutral':
            return self._find_delta_neutral_straddle(options_chain)
        elif method == 'interpolated':
            return self._find_interpolated_straddle(options_chain)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _find_closest_strike_straddle(self, options_chain: OptionsChain) -> Optional[ATMStraddle]:
        """Find straddle at strike closest to spot price"""
        
        # Find available strikes in both calls and puts
        call_strikes = set(options_chain.calls['strike'].values)
        put_strikes = set(options_chain.puts['strike'].values)
        common_strikes = call_strikes.intersection(put_strikes)
        
        if not common_strikes:
            return None
        
        # Find closest strike to spot
        closest_strike = min(common_strikes, key=lambda x: abs(x - options_chain.spot_price))
        
        # Get call and put at this strike
        call_data = options_chain.calls[options_chain.calls['strike'] == closest_strike].iloc[0]
        put_data = options_chain.puts[options_chain.puts['strike'] == closest_strike].iloc[0]
        
        # Calculate straddle metrics
        total_premium = call_data['last_price'] + put_data['last_price']
        
        # Calculate combined implied volatility (weighted average)
        call_iv = call_data.get('implied_vol')
        put_iv = put_data.get('implied_vol')
        
        if call_iv is not None and put_iv is not None:
            # Weight by option values
            call_weight = call_data['last_price'] / total_premium
            put_weight = put_data['last_price'] / total_premium
            combined_iv = call_weight * call_iv + put_weight * put_iv
        else:
            combined_iv = None
        
        # Calculate days to expiry
        if options_chain.timestamp:
            time_diff = options_chain.expiration - options_chain.timestamp
        else:
            time_diff = options_chain.expiration - datetime.now()
        days_to_expiry = time_diff.days
        
        return ATMStraddle(
            strike=closest_strike,
            call_price=call_data['last_price'],
            put_price=put_data['last_price'],
            total_premium=total_premium,
            implied_vol=combined_iv,
            days_to_expiry=days_to_expiry,
            delta_neutral_strike=closest_strike  # Approximation
        )
    
    def _find_delta_neutral_straddle(self, options_chain: OptionsChain) -> Optional[ATMStraddle]:
        """Find delta-neutral straddle (where call delta + put delta = 0)"""
        # This is a more sophisticated approach that would require delta calculations
        # For now, fall back to closest strike method
        logger.warning("Delta-neutral straddle calculation not fully implemented, using closest strike")
        return self._find_closest_strike_straddle(options_chain)
    
    def _find_interpolated_straddle(self, options_chain: OptionsChain) -> Optional[ATMStraddle]:
        """Create synthetic ATM straddle via interpolation"""
        # This would interpolate between nearby strikes to create exact ATM straddle
        # For now, fall back to closest strike method
        logger.warning("Interpolated straddle calculation not fully implemented, using closest strike")
        return self._find_closest_strike_straddle(options_chain)

class ImpliedMoveComputer:
    """Compute implied moves from straddle and IV data"""
    
    def __init__(self):
        self.calculation_methods = ['straddle_premium', 'black_scholes', 'hybrid']
    
    def compute_implied_move(self, atm_straddle: ATMStraddle, 
                           method: str = 'hybrid') -> ImpliedMove:
        """
        Compute implied move from ATM straddle
        
        Args:
            atm_straddle: ATM straddle data
            method: Calculation method ('straddle_premium', 'black_scholes', 'hybrid')
        """
        if method == 'straddle_premium':
            return self._compute_from_premium(atm_straddle)
        elif method == 'black_scholes':
            return self._compute_from_iv(atm_straddle)
        elif method == 'hybrid':
            return self._compute_hybrid(atm_straddle)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_from_premium(self, straddle: ATMStraddle) -> ImpliedMove:
        """Compute move directly from straddle premium"""
        
        # The straddle premium represents the market's expectation of movement
        # This is approximately a 1-sigma move (68% probability)
        straddle_move = straddle.total_premium
        straddle_percent = (straddle_move / straddle.strike) * 100
        
        # Scale to different sigma levels
        # Straddle ~ 0.8 sigma empirically, so adjust
        sigma_scaling = 1.25  # Empirical scaling factor
        one_sigma_move = straddle_move * sigma_scaling
        one_sigma_percent = (one_sigma_move / straddle.strike) * 100
        
        two_sigma_move = one_sigma_move * 2
        two_sigma_percent = (two_sigma_move / straddle.strike) * 100
        
        # Confidence interval (1 sigma)
        lower_bound = straddle.strike - one_sigma_move
        upper_bound = straddle.strike + one_sigma_move
        
        return ImpliedMove(
            one_sigma_move=one_sigma_move,
            one_sigma_percent=one_sigma_percent,
            two_sigma_move=two_sigma_move,
            two_sigma_percent=two_sigma_percent,
            straddle_move=straddle_move,
            straddle_percent=straddle_percent,
            confidence_interval=(lower_bound, upper_bound),
            calculation_method='straddle_premium',
            metadata={
                'straddle_strike': straddle.strike,
                'straddle_cost': straddle.total_premium,
                'days_to_expiry': straddle.days_to_expiry,
                'scaling_factor': sigma_scaling
            }
        )
    
    def _compute_from_iv(self, straddle: ATMStraddle) -> ImpliedMove:
        """Compute move from Black-Scholes implied volatility"""
        
        if straddle.implied_vol is None:
            logger.warning("No implied volatility available, falling back to premium method")
            return self._compute_from_premium(straddle)
        
        # Convert days to years
        time_to_expiry = straddle.days_to_expiry / 365.25
        
        if time_to_expiry <= 0:
            raise ValueError("Invalid time to expiry")
        
        # Calculate moves using vol * sqrt(time)
        vol_term = straddle.implied_vol * np.sqrt(time_to_expiry)
        
        one_sigma_move = straddle.strike * vol_term
        one_sigma_percent = vol_term * 100
        
        two_sigma_move = straddle.strike * (2 * vol_term)
        two_sigma_percent = (2 * vol_term) * 100
        
        # Straddle approximation (usually ~0.8 sigma)
        straddle_move = one_sigma_move * 0.8
        straddle_percent = one_sigma_percent * 0.8
        
        # Confidence interval
        lower_bound = straddle.strike * (1 - vol_term)
        upper_bound = straddle.strike * (1 + vol_term)
        
        return ImpliedMove(
            one_sigma_move=one_sigma_move,
            one_sigma_percent=one_sigma_percent,
            two_sigma_move=two_sigma_move,
            two_sigma_percent=two_sigma_percent,
            straddle_move=straddle_move,
            straddle_percent=straddle_percent,
            confidence_interval=(lower_bound, upper_bound),
            calculation_method='black_scholes',
            metadata={
                'implied_vol': straddle.implied_vol,
                'time_to_expiry': time_to_expiry,
                'vol_term': vol_term,
                'straddle_strike': straddle.strike
            }
        )
    
    def _compute_hybrid(self, straddle: ATMStraddle) -> ImpliedMove:
        """Hybrid approach combining premium and IV methods"""
        
        try:
            # Try IV method first
            iv_result = self._compute_from_iv(straddle)
            premium_result = self._compute_from_premium(straddle)
            
            # Blend the results (weighted average)
            iv_weight = 0.7 if straddle.implied_vol is not None else 0.0
            premium_weight = 1.0 - iv_weight
            
            one_sigma_move = (iv_weight * iv_result.one_sigma_move + 
                             premium_weight * premium_result.one_sigma_move)
            one_sigma_percent = (one_sigma_move / straddle.strike) * 100
            
            two_sigma_move = one_sigma_move * 2
            two_sigma_percent = one_sigma_percent * 2
            
            # Use premium method for straddle move (more direct)
            straddle_move = premium_result.straddle_move
            straddle_percent = premium_result.straddle_percent
            
            # Confidence interval
            lower_bound = straddle.strike - one_sigma_move
            upper_bound = straddle.strike + one_sigma_move
            
            return ImpliedMove(
                one_sigma_move=one_sigma_move,
                one_sigma_percent=one_sigma_percent,
                two_sigma_move=two_sigma_move,
                two_sigma_percent=two_sigma_percent,
                straddle_move=straddle_move,
                straddle_percent=straddle_percent,
                confidence_interval=(lower_bound, upper_bound),
                calculation_method='hybrid',
                metadata={
                    'iv_weight': iv_weight,
                    'premium_weight': premium_weight,
                    'iv_result': iv_result.metadata if straddle.implied_vol else None,
                    'premium_result': premium_result.metadata
                }
            )
            
        except Exception as e:
            logger.warning(f"Hybrid calculation failed: {e}, falling back to premium method")
            return self._compute_from_premium(straddle)

class OptionsDataProcessor:
    """Process and validate options data for implied move calculations"""
    
    def __init__(self):
        self.iv_calculator = ImpliedVolatilityCalculator()
        self.straddle_finder = ATMStraddleFinder(self.iv_calculator)
        self.move_computer = ImpliedMoveComputer()
    
    def process_options_chain(self, options_data: Dict, 
                            spot_price: float,
                            expiration: datetime,
                            risk_free_rate: float = 0.02) -> OptionsChain:
        """
        Process raw options data into OptionsChain format
        
        Args:
            options_data: Dict with 'calls' and 'puts' DataFrames
            spot_price: Current underlying price
            expiration: Option expiration date
            risk_free_rate: Risk-free rate for calculations
        """
        
        # Validate and clean data
        calls_df = self._validate_options_data(options_data.get('calls', pd.DataFrame()))
        puts_df = self._validate_options_data(options_data.get('puts', pd.DataFrame()))
        
        # Create options chain
        options_chain = OptionsChain(
            calls=calls_df,
            puts=puts_df,
            spot_price=spot_price,
            expiration=expiration,
            risk_free_rate=risk_free_rate,
            timestamp=datetime.now()
        )
        
        # Calculate implied volatilities
        options_chain = self.iv_calculator.calculate_chain_ivs(options_chain)
        
        return options_chain
    
    def _validate_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean options data"""
        
        if df.empty:
            return df
        
        # Required columns
        required_cols = ['strike', 'last_price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data
        df = df.copy()
        
        # Remove rows with invalid data
        df = df[df['strike'] > 0]
        df = df[df['last_price'] > 0]
        df = df.dropna(subset=required_cols)
        
        # Sort by strike
        df = df.sort_values('strike').reset_index(drop=True)
        
        return df
    
    def compute_implied_moves_pipeline(self, options_data: Dict,
                                     spot_price: float,
                                     expiration: datetime,
                                     risk_free_rate: float = 0.02) -> Tuple[ATMStraddle, ImpliedMove]:
        """
        Complete pipeline to compute implied moves from raw options data
        
        Returns:
            Tuple of (ATMStraddle, ImpliedMove)
        """
        
        # Process options chain
        options_chain = self.process_options_chain(
            options_data, spot_price, expiration, risk_free_rate
        )
        
        # Find ATM straddle
        atm_straddle = self.straddle_finder.find_atm_straddle(options_chain)
        if atm_straddle is None:
            raise ValueError("Could not find suitable ATM straddle")
        
        # Compute implied move
        implied_move = self.move_computer.compute_implied_move(atm_straddle)
        
        return atm_straddle, implied_move