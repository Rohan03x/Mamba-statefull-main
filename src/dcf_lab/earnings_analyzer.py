"""
HEDGE-FUND GRADE: Earnings Analyzer - Event & Quality Signals

Extracts 14-16 earnings-related features from EODHD fundamentals data:

A. Core Earnings (2-4):
   - eps_surprise_pct: (actual - estimate) / estimate * 100
   - revenue_surprise_pct: Revenue surprise percentage
   [OPTIONAL: eps_surprise_z, revenue_surprise_z (rolling 3-5y normalization)]
   REMOVED: eps_actual, eps_estimate, revenue_actual, revenue_estimate
   Reason: Scale issues, not cross-sectionally comparable, encourage memorization

B. Growth Trends (4):
   - eps_growth_qoq: Quarter-over-quarter EPS growth (winsorized ±200%)
   - eps_growth_yoy: Year-over-year EPS growth (winsorized ±200%)
   - revenue_growth_qoq: Quarter-over-quarter revenue growth (winsorized ±200%)
   - revenue_growth_yoy: Year-over-year revenue growth (winsorized ±200%)

C. Patterns (2):
   - earnings_beat_streak: Consecutive quarters of beating estimates
   - earnings_miss_streak: Consecutive quarters of missing estimates
   REMOVED: earnings_volatility_flag (binary → continuous)
   ADDED: earnings_surprise_volatility (rolling std of surprise_pct)

D. Beat Consistency (2) - HIGH ALPHA:
   - beat_streak: Consecutive quarters of beats (extended)
   - beat_rate_3y: Percentage of beats over last 12 quarters

E. Revision Breadth & Dispersion (2-3) - CRITICAL ALPHA:
   - revision_breadth: % of analysts raising vs cutting estimates (proxy)
   - estimate_dispersion: Standard deviation of EPS estimates (coefficient of variation)
   [OPTIONAL: revision_breadth_change (acceleration)]

F. Event Timing (2):
   - days_since_earnings: Days since last earnings report
   - earnings_event_decay: exp(-days_since_earnings / tau) where tau≈10-15
   (post-earnings drift: react strongly after, gradually forget)

G. Anticipation (1):
   - days_to_next_earnings: Days until next expected earnings
   (pre-earnings positioning, volatility rises before events)

Total: 14-16 features (optimal size for earnings family)

Data Source: EODHD Fundamentals API
Update Frequency: Quarterly (within days of earnings release)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EarningsAnalyzer:
    """Extract hedge-fund grade earnings features from EODHD fundamentals"""
    
    def __init__(self):
        self.growth_clip = 200.0  # Winsorize growth to ±200%
        self.event_decay_tau = 12.0  # Trading days for exponential decay (10-15 recommended)
        self.quarterly_days = 91  # Approximate days per quarter for next earnings estimate
    
    def get_earnings_features(self, ticker: str) -> Dict[str, float]:
        """
        Extract 14-16 hedge-fund grade earnings features for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with 14-16 earnings features (raw levels removed)
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔍 Fetching earnings data for {ticker}")
            
            eodhd = get_eodhd_provider()
            fundamentals = eodhd.get_fundamentals(ticker)
            
            if not fundamentals:
                logger.warning(f"No fundamentals data for {ticker}")
                return {}
            
            # Extract earnings and financials
            earnings_data = fundamentals.get('Earnings', {})
            financials_data = fundamentals.get('Financials', {})
            
            earnings_history = earnings_data.get('History', {})
            income_stmt = financials_data.get('Income_Statement', {})
            quarterly_financials = income_stmt.get('quarterly', {})
            
            if not earnings_history:
                logger.warning(f"No earnings history for {ticker}")
                return {}
            
            # Convert to sorted list (newest first)
            earnings_list = [
                {'date': date, **report}
                for date, report in earnings_history.items()
                if report.get('epsActual') is not None  # Only actual reports
            ]
            earnings_list.sort(key=lambda x: x['date'], reverse=True)
            
            if len(earnings_list) < 1:
                logger.warning(f"No actual earnings reports for {ticker}")
                return {}
            
            # Get revenue data from quarterly financials
            revenue_list = [
                {'date': date, 'revenue': float(stmt.get('totalRevenue', 0))}
                for date, stmt in quarterly_financials.items()
            ]
            revenue_list.sort(key=lambda x: x['date'], reverse=True)
            
            # Generate features
            features = {}
            
            # ===== A. CORE EARNINGS FEATURES (2-4) =====
            # HEDGE-FUND: Remove raw levels, keep only surprises
            latest_earnings = earnings_list[0]
            
            # Get raw values for calculation (NOT returned as features)
            eps_actual = float(latest_earnings.get('epsActual', 0))
            eps_estimate = float(latest_earnings.get('epsEstimate', 0))
            
            # 1. eps_surprise_pct (KEEP)
            if eps_estimate and eps_estimate != 0:
                features['eps_surprise_pct'] = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
            else:
                features['eps_surprise_pct'] = 0.0
            
            # 2-3. Revenue surprise (KEEP)
            if revenue_list:
                latest_revenue = revenue_list[0]['revenue']
                
                # Estimate revenue estimate (use average of past 4 quarters as proxy)
                if len(revenue_list) >= 5:
                    past_revenues = [r['revenue'] for r in revenue_list[1:5]]
                    avg_revenue = np.mean(past_revenues)
                    
                    # Revenue surprise
                    if avg_revenue > 0:
                        features['revenue_surprise_pct'] = ((latest_revenue - avg_revenue) / avg_revenue) * 100
                    else:
                        features['revenue_surprise_pct'] = 0.0
                else:
                    features['revenue_surprise_pct'] = 0.0
            else:
                features['revenue_surprise_pct'] = 0.0
            
            # REMOVED: eps_actual, eps_estimate, revenue_actual, revenue_estimate
            # Reason: Scale issues, not cross-sectionally comparable
            
            # ===== B. GROWTH TRENDS (4) =====
            # HEDGE-FUND: Winsorize to ±200% (growth explodes after losses/small bases)
            
            # 4. eps_growth_qoq (quarter-over-quarter)
            if len(earnings_list) >= 2:
                prev_eps = float(earnings_list[1].get('epsActual', 0))
                if prev_eps != 0:
                    growth_qoq = ((eps_actual - prev_eps) / abs(prev_eps)) * 100
                    features['eps_growth_qoq'] = np.clip(growth_qoq, -self.growth_clip, self.growth_clip)
                else:
                    features['eps_growth_qoq'] = 0.0
            else:
                features['eps_growth_qoq'] = 0.0
            
            # 5. eps_growth_yoy (year-over-year)
            if len(earnings_list) >= 5:  # 4 quarters ago
                yoy_eps = float(earnings_list[4].get('epsActual', 0))
                if yoy_eps != 0:
                    growth_yoy = ((eps_actual - yoy_eps) / abs(yoy_eps)) * 100
                    features['eps_growth_yoy'] = np.clip(growth_yoy, -self.growth_clip, self.growth_clip)
                else:
                    features['eps_growth_yoy'] = 0.0
            else:
                features['eps_growth_yoy'] = 0.0
            
            # 6. revenue_growth_qoq
            if len(revenue_list) >= 2:
                prev_revenue = revenue_list[1]['revenue']
                curr_revenue = revenue_list[0]['revenue']
                if prev_revenue != 0:
                    growth_qoq = ((curr_revenue - prev_revenue) / prev_revenue) * 100
                    features['revenue_growth_qoq'] = np.clip(growth_qoq, -self.growth_clip, self.growth_clip)
                else:
                    features['revenue_growth_qoq'] = 0.0
            else:
                features['revenue_growth_qoq'] = 0.0
            
            # 7. revenue_growth_yoy
            if len(revenue_list) >= 5:
                yoy_revenue = revenue_list[4]['revenue']
                curr_revenue = revenue_list[0]['revenue']
                if yoy_revenue != 0:
                    growth_yoy = ((curr_revenue - yoy_revenue) / yoy_revenue) * 100
                    features['revenue_growth_yoy'] = np.clip(growth_yoy, -self.growth_clip, self.growth_clip)
                else:
                    features['revenue_growth_yoy'] = 0.0
            else:
                features['revenue_growth_yoy'] = 0.0
            
            # ===== C. PATTERNS (2) =====
            # HEDGE-FUND: Remove binary volatility_flag, add continuous surprise_volatility
            
            # 8. earnings_beat_streak (last 3 quarters)
            beat_streak = 0
            for i in range(min(3, len(earnings_list))):
                report = earnings_list[i]
                surprise = report.get('surprisePercent')
                if surprise is not None and float(surprise) > 0:
                    beat_streak += 1
                else:
                    break  # Streak broken
            features['earnings_beat_streak'] = beat_streak
            
            # 9. earnings_miss_streak
            miss_streak = 0
            for i in range(min(3, len(earnings_list))):
                report = earnings_list[i]
                surprise = report.get('surprisePercent')
                if surprise is not None and float(surprise) < 0:
                    miss_streak += 1
                else:
                    break  # Streak broken
            features['earnings_miss_streak'] = miss_streak
            
            # 10. earnings_surprise_volatility (NEW: replaces binary volatility_flag)
            # Rolling standard deviation of surprise percentages (continuous uncertainty metric)
            if len(earnings_list) >= 8:
                recent_surprises = [
                    float(e.get('surprisePercent', 0)) for e in earnings_list[:8]
                    if e.get('surprisePercent') is not None
                ]
                if len(recent_surprises) >= 4:
                    features['earnings_surprise_volatility'] = np.std(recent_surprises)
                else:
                    features['earnings_surprise_volatility'] = 0.0
            else:
                features['earnings_surprise_volatility'] = 0.0
            
            # REMOVED: earnings_volatility_flag (binary flag → continuous volatility)
            
            # ===== D. BEAT CONSISTENCY (2) - HIGH ALPHA =====
            # NO CHANGES - these are excellent features
            
            # 11. beat_streak (extended version - all consecutive beats)
            beat_streak_extended = 0
            for report in earnings_list:
                surprise = report.get('surprisePercent')
                if surprise is not None and float(surprise) > 0:
                    beat_streak_extended += 1
                else:
                    break  # Streak broken
            features['beat_streak'] = beat_streak_extended
            
            # 12. beat_rate_3y (beats / total over last 12 quarters)
            if len(earnings_list) >= 12:
                beats_12q = sum(
                    1 for e in earnings_list[:12]
                    if e.get('surprisePercent') is not None and float(e.get('surprisePercent', 0)) > 0
                )
                features['beat_rate_3y'] = (beats_12q / 12.0) * 100  # Percentage
            elif len(earnings_list) >= 4:  # At least 1 year
                beats = sum(
                    1 for e in earnings_list
                    if e.get('surprisePercent') is not None and float(e.get('surprisePercent', 0)) > 0
                )
                features['beat_rate_3y'] = (beats / len(earnings_list)) * 100
            else:
                features['beat_rate_3y'] = 0.0
            
            # ===== E. REVISION BREADTH & DISPERSION (2-3) - CRITICAL ALPHA =====
            # NO CHANGES - these are excellent features
            
            # 13. revision_breadth (proxy: trend in EPS estimates over recent quarters)
            # Since we don't have analyst-level data, use estimate momentum as proxy
            # Positive momentum = estimates rising (breadth positive)
            if len(earnings_list) >= 4:
                recent_estimates = [
                    float(e.get('epsEstimate', 0)) for e in earnings_list[:4]
                    if e.get('epsEstimate') is not None
                ]
                if len(recent_estimates) >= 2:
                    # Count quarters where estimate increased vs decreased
                    increases = sum(
                        1 for i in range(len(recent_estimates) - 1)
                        if recent_estimates[i] > recent_estimates[i+1]
                    )
                    total_changes = len(recent_estimates) - 1
                    # Convert to -100 to +100 scale (% raising - % cutting)
                    features['revision_breadth'] = ((increases / total_changes) * 200) - 100
                else:
                    features['revision_breadth'] = 0.0
            else:
                features['revision_breadth'] = 0.0
            
            # 14. estimate_dispersion (coefficient of variation of recent surprises)
            # Proxy: volatility of surprise percentages over last 8 quarters
            if len(earnings_list) >= 8:
                recent_surprises = [
                    float(e.get('surprisePercent', 0)) for e in earnings_list[:8]
                    if e.get('surprisePercent') is not None
                ]
                if len(recent_surprises) >= 4:
                    surprise_mean = np.mean(np.abs(recent_surprises))  # Mean absolute surprise
                    surprise_std = np.std(recent_surprises)
                    
                    if surprise_mean > 0:
                        features['estimate_dispersion'] = (surprise_std / surprise_mean) * 100
                    else:
                        features['estimate_dispersion'] = 0.0
                else:
                    features['estimate_dispersion'] = 0.0
            else:
                features['estimate_dispersion'] = 0.0
            
            # ===== F. EVENT TIMING (2) =====
            # HEDGE-FUND: Add exponential decay for post-earnings drift
            
            # 15. days_since_earnings
            latest_date = latest_earnings['date']
            try:
                earnings_date = pd.to_datetime(latest_date)
                current_date = pd.Timestamp.now()
                days_since = (current_date - earnings_date).days
                features['days_since_earnings'] = days_since
                
                # 16. earnings_event_decay (NEW)
                # exp(-days_since_earnings / tau) where tau ≈ 10-15 trading days
                # React strongly just after earnings, gradually forget
                features['earnings_event_decay'] = np.exp(-days_since / self.event_decay_tau)
            except Exception as e:
                logger.debug(f"Could not calculate days_since_earnings: {e}")
                features['days_since_earnings'] = 0.0
                features['earnings_event_decay'] = 0.0
            
            # ===== G. ANTICIPATION (1) =====
            # HEDGE-FUND: Pre-earnings positioning (pairs with cboe_term/correlation vol)
            
            # 17. days_to_next_earnings (NEW)
            # Estimate: assume quarterly earnings (91 days) from last report
            # More sophisticated version would use actual earnings calendar
            try:
                next_earnings_estimate = earnings_date + pd.Timedelta(days=self.quarterly_days)
                days_to_next = (next_earnings_estimate - current_date).days
                # Clip to reasonable range (0 to 120 days)
                features['days_to_next_earnings'] = max(0, min(120, days_to_next))
            except Exception as e:
                logger.debug(f"Could not calculate days_to_next_earnings: {e}")
                features['days_to_next_earnings'] = 0.0
            
            logger.info(f"✅ Generated {len(features)} hedge-fund grade earnings features for {ticker}")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error generating earnings features for {ticker}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}

    def get_earnings_timeseries(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Build a daily earnings feature time series.

        The model is event-driven: compute features on earnings report dates and
        forward-fill until the next report. This avoids the degenerate behavior of
        broadcasting a single "latest" snapshot across multi-year windows.
        """
        from src.data_sources.eodhd_provider import get_eodhd_provider

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        eodhd = get_eodhd_provider()
        fundamentals = eodhd.get_fundamentals(ticker)
        if not fundamentals:
            return pd.DataFrame()

        earnings_history = (fundamentals.get('Earnings', {}) or {}).get('History', {}) or {}
        if not earnings_history:
            return pd.DataFrame()

        financials_data = fundamentals.get('Financials', {}) or {}
        income_stmt = (financials_data.get('Income_Statement', {}) or {})
        quarterly_financials = (income_stmt.get('quarterly', {}) or {})

        # Build a sorted quarterly revenue series (oldest -> newest). EODHD uses
        # statement period end dates; we'll align each earnings report to the
        # most recent statement date on or before the report.
        revenue_points: List[dict] = []
        for date_str, stmt in quarterly_financials.items():
            if stmt is None:
                continue
            raw_rev = stmt.get('totalRevenue')
            if raw_rev is None:
                continue
            try:
                dt = pd.to_datetime(date_str)
            except Exception:
                continue
            try:
                rev_val = float(raw_rev)
            except Exception:
                continue
            revenue_points.append({'date': dt, 'revenue': rev_val})

        revenue_points.sort(key=lambda x: x['date'])
        revenue_dates = np.array([p['date'].to_datetime64() for p in revenue_points], dtype='datetime64[ns]')
        revenue_vals = np.array([float(p['revenue']) for p in revenue_points], dtype=float) if revenue_points else np.array([], dtype=float)

        # Build sorted list (oldest -> newest) so we can compute lagged features.
        reports: List[dict] = []
        for date_str, report in earnings_history.items():
            if report is None:
                continue
            if report.get('epsActual') is None:
                continue
            try:
                dt = pd.to_datetime(date_str)
            except Exception:
                continue
            reports.append({'date': dt, **report})

        if not reports:
            return pd.DataFrame()

        reports.sort(key=lambda x: x['date'])

        # Pre-extract sequences for vector-friendly access
        dates = [r['date'] for r in reports]
        eps_actual = np.array([float(r.get('epsActual', 0) or 0) for r in reports], dtype=float)
        eps_est = np.array([float(r.get('epsEstimate', 0) or 0) for r in reports], dtype=float)
        surprise_pct = np.array([
            float(r.get('surprisePercent', 0) or 0) if r.get('surprisePercent') is not None else 0.0
            for r in reports
        ], dtype=float)

        # Align revenue to earnings report dates (best-effort).
        revenue_aligned = np.zeros(len(dates), dtype=float)
        if revenue_points:
            report_dates64 = np.array([pd.to_datetime(d).to_datetime64() for d in dates], dtype='datetime64[ns]')
            # For each report date, pick the latest revenue statement date <= report date.
            pos = np.searchsorted(revenue_dates, report_dates64, side='right') - 1
            pos = np.clip(pos, 0, max(0, len(revenue_vals) - 1))
            revenue_aligned = revenue_vals[pos]

        # Event-level features
        rows: List[dict] = []
        beat_streak = 0
        miss_streak = 0
        for i, dt in enumerate(dates):
            est = eps_est[i]
            act = eps_actual[i]
            eps_surprise = ((act - est) / abs(est) * 100.0) if est not in (0.0, -0.0) else 0.0

            is_beat = 1.0 if surprise_pct[i] > 0 else 0.0
            is_miss = 1.0 if surprise_pct[i] < 0 else 0.0
            beat_streak = beat_streak + 1 if is_beat else 0
            miss_streak = miss_streak + 1 if is_miss else 0

            # QoQ / YoY on EPS actual
            eps_growth_qoq = 0.0
            if i >= 1 and eps_actual[i - 1] != 0:
                eps_growth_qoq = np.clip(((act - eps_actual[i - 1]) / abs(eps_actual[i - 1])) * 100.0,
                                         -self.growth_clip, self.growth_clip)

            eps_growth_yoy = 0.0
            if i >= 4 and eps_actual[i - 4] != 0:
                eps_growth_yoy = np.clip(((act - eps_actual[i - 4]) / abs(eps_actual[i - 4])) * 100.0,
                                         -self.growth_clip, self.growth_clip)

            # Revenue QoQ / YoY (from quarterly income statement when available)
            revenue_growth_qoq = 0.0
            if i >= 1 and revenue_aligned[i - 1] != 0:
                revenue_growth_qoq = np.clip(((revenue_aligned[i] - revenue_aligned[i - 1]) / revenue_aligned[i - 1]) * 100.0,
                                             -self.growth_clip, self.growth_clip)

            revenue_growth_yoy = 0.0
            if i >= 4 and revenue_aligned[i - 4] != 0:
                revenue_growth_yoy = np.clip(((revenue_aligned[i] - revenue_aligned[i - 4]) / revenue_aligned[i - 4]) * 100.0,
                                             -self.growth_clip, self.growth_clip)

            # Revenue surprise proxy: compare current revenue to avg of prior 4 quarters.
            revenue_surprise_pct = 0.0
            if i >= 4:
                past = revenue_aligned[i - 4:i]
                avg_rev = float(np.mean(past)) if len(past) else 0.0
                if avg_rev != 0:
                    revenue_surprise_pct = ((revenue_aligned[i] - avg_rev) / avg_rev) * 100.0

            # Rolling beat rate over last 12 reports
            win = surprise_pct[max(0, i - 11): i + 1]
            beat_rate_3y = float((win > 0).mean() * 100.0) if len(win) else 0.0

            # Surprise volatility over last 8 reports
            win2 = surprise_pct[max(0, i - 7): i + 1]
            earnings_surprise_volatility = float(np.std(win2)) if len(win2) >= 4 else 0.0

            # Revision breadth proxy: estimate momentum over last 4 reports
            win_est = eps_est[max(0, i - 3): i + 1]
            revision_breadth = 0.0
            if len(win_est) >= 2:
                # count increases going forward (rising estimates)
                increases = sum(1 for j in range(len(win_est) - 1) if win_est[j] > win_est[j + 1])
                total = len(win_est) - 1
                revision_breadth = ((increases / total) * 200.0) - 100.0

            # Estimate dispersion proxy: coefficient of variation of recent surprise %.
            estimate_dispersion = 0.0
            win_disp = surprise_pct[max(0, i - 7): i + 1]
            if len(win_disp) >= 4:
                mean_abs = float(np.mean(np.abs(win_disp)))
                std = float(np.std(win_disp))
                if mean_abs > 0:
                    estimate_dispersion = (std / mean_abs) * 100.0

            rows.append({
                'date': dt,
                'has_data': 1.0,
                'eps_surprise_pct': float(eps_surprise),
                'revenue_surprise_pct': float(revenue_surprise_pct),
                'surprise_percent': float(surprise_pct[i]),
                'eps_growth_qoq': float(eps_growth_qoq),
                'eps_growth_yoy': float(eps_growth_yoy),
                'revenue_growth_qoq': float(revenue_growth_qoq),
                'revenue_growth_yoy': float(revenue_growth_yoy),
                'earnings_beat_streak': float(beat_streak),
                'earnings_miss_streak': float(miss_streak),
                'earnings_surprise_volatility': float(earnings_surprise_volatility),
                'beat_rate_3y': float(beat_rate_3y),
                'revision_breadth': float(revision_breadth),
                'estimate_dispersion': float(estimate_dispersion),
            })

        event_df = pd.DataFrame(rows)
        event_df['date'] = pd.to_datetime(event_df['date'])
        event_df = event_df.set_index('date').sort_index()

        # Daily series: ffill between earnings events; bfill to cover leading edge.
        daily_index = pd.date_range(start=start_ts, end=end_ts, freq='D')
        daily = event_df.reindex(daily_index).ffill().bfill()
        daily.index.name = 'date'

        # days_since_earnings / decay based on last earnings date
        last_event_dates = pd.Series(event_df.index, index=event_df.index)
        last_event_for_day = last_event_dates.reindex(daily_index).ffill().bfill()
        days_since = (pd.Series(daily_index, index=daily_index) - last_event_for_day).dt.days.astype(float)
        daily['days_since_earnings'] = days_since
        daily['earnings_event_decay'] = np.exp(-days_since / float(self.event_decay_tau))

        # days_to_next_earnings based on next earnings date
        event_dates = np.array(event_df.index.values, dtype='datetime64[ns]')
        day_values = np.array(daily_index.values, dtype='datetime64[ns]')
        pos = np.searchsorted(event_dates, day_values, side='right')
        next_pos = np.clip(pos, 0, len(event_dates) - 1)
        next_event = event_dates[next_pos]
        days_to_next = (next_event - day_values).astype('timedelta64[D]').astype(float)
        # If the day is beyond the last known report, treat as unknown (0)
        days_to_next[pos >= len(event_dates)] = 0.0
        daily['days_to_next_earnings'] = np.clip(days_to_next, 0.0, 120.0)

        daily = daily.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return daily


def get_earnings_analyzer() -> EarningsAnalyzer:
    """Get singleton earnings analyzer instance"""
    return EarningsAnalyzer()
