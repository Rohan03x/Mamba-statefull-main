from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf


class YFinanceProvider:
    """Dev provider using yfinance as a convenient data source."""

    def get_profile(self, ticker: str) -> Dict:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
        return {
            "ticker": ticker,
            "name": info.get("shortName") or "",
            "country": info.get("country") or None,
            "sector": info.get("sector") or None,
            "currency": info.get("currency") or "USD",
        }

    def get_market(self, ticker: str) -> Dict:
        from ..utils import sanitize_dt_index
        
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
        hist = t.history(period="1d")
        
        # Global datetime sanitation: make timestamps tz-naive UTC
        if not hist.empty:
            hist = sanitize_dt_index(hist)
        
        price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        return {
            "ticker": ticker,
            "price": price,
            "shares": info.get("sharesOutstanding"),
            "marketcap": info.get("marketCap"),
            "beta": info.get("beta"),
        }

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        from ..utils import sanitize_dt_index
        
        t = yf.Ticker(ticker)
        income = t.financials.T.reset_index().rename(columns={"index": "date"})
        balance = t.balance_sheet.T.reset_index().rename(
            columns={"index": "date"})
        cash = t.cashflow.T.reset_index().rename(columns={"index": "date"})
        for df in (income, balance, cash):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df.sort_values("date", inplace=True)
                # Global datetime sanitation: make timestamps tz-naive UTC
                df = sanitize_dt_index(df, col="date")
                df = df.tail(years)
        return {
            "income": income.to_dict(orient="records"),
            "balance": balance.to_dict(orient="records"),
            "cashflow": cash.to_dict(orient="records"),
        }

    def get_estimates(self, ticker: str) -> Dict:
        """Get analyst estimates with basic implementation."""
        try:
            # Basic implementation using yfinance recommendations
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            estimates = {}
            
            # Extract basic estimates from ticker info
            if 'targetMeanPrice' in info:
                estimates['target_price'] = info['targetMeanPrice']
            if 'recommendationMean' in info:
                estimates['recommendation_mean'] = info['recommendationMean']
            if 'numberOfAnalystOpinions' in info:
                estimates['analyst_count'] = info['numberOfAnalystOpinions']
                
            return estimates
            
        except Exception:
            # Return empty dict if data unavailable
            return {}

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        """Get news with multiple fallback sources."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Try different sources in order of preference
        items = self._get_yfinance_news(ticker, cutoff)
        if not items:
            items = self._get_yahoo_rss_news(ticker, cutoff)
        if not items:
            items = self._get_finnhub_news(ticker, cutoff, days)
        
        return items
    
    def _get_yfinance_news(self, ticker: str, cutoff: datetime) -> List[Dict]:
        """Get news from yfinance API."""
        items = []
        try:
            t = yf.Ticker(ticker)
            news = getattr(t, "news", []) or []
            for n in news:
                ts = n.get("providerPublishTime")
                # Filter by cutoff when epoch is available
                if isinstance(ts, (int, float)):
                    if datetime.fromtimestamp(ts, tz=timezone.utc) < cutoff:
                        continue
                items.append({
                    "date": ts,
                    "headline": n.get("title"),
                    "source": n.get("provider"),
                    "url": n.get("link"),
                    "tickers": [ticker],
                })
        except Exception:
            pass
        return items
    
    def _get_yahoo_rss_news(self, ticker: str, cutoff: datetime) -> List[Dict]:
        """Get news from Yahoo RSS feed."""
        items = []
        try:
            url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline?"
                   f"s={ticker}&region=US&lang=en-US")
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for it in root.findall('.//item'):
                title = (it.findtext('title') or '').strip()
                link = (it.findtext('link') or '').strip()
                pub = it.findtext('pubDate') or ''
                
                # Parse RFC822 date
                dt = self._parse_rfc822_date(pub)
                if dt and dt < cutoff:
                    continue
                    
                items.append({
                    "date": int(dt.timestamp()) if dt else None,
                    "headline": title,
                    "source": "Yahoo Finance",
                    "url": link,
                    "tickers": [ticker],
                })
        except Exception:
            pass
        return items
    
    def _get_finnhub_news(self, ticker: str, cutoff: datetime,
                          days: int) -> List[Dict]:
        """Get news from Finnhub API."""
        items = []
        try:
            token = os.environ.get("FINNHUB_TOKEN")
            if not token:
                return items
                
            to_dt = datetime.now(datetime.timezone.utc).date()
            from_dt = to_dt - timedelta(days=days)
            fn = "https://finnhub.io/api/v1/company-news"
            rr = requests.get(
                fn,
                params={
                    "symbol": ticker,
                    "from": from_dt.isoformat(),
                    "to": to_dt.isoformat(),
                    "token": token},
                timeout=20)
            rr.raise_for_status()
            data = rr.json() or []
            
            for n in data:
                ts = n.get("datetime")
                if ts:
                    dt = datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                    if dt < cutoff:
                        continue
                else:
                    dt = None
                    
                items.append({
                    "date": ts,
                    "headline": n.get("headline"),
                    "source": n.get("source"),
                    "url": n.get("url"),
                    "tickers": [ticker],
                })
        except Exception:
            pass
        return items
    
    def _parse_rfc822_date(self, date_str: str) -> Optional[datetime]:
        """Parse RFC822 date string."""
        try:
            return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
        except Exception:
            return None
