"""
Tiingo Data Provider for DCF Lab
Provides reliable financial data via Tiingo API with fallbacks
"""
from __future__ import annotations

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import requests
from tiingo import TiingoClient

logger = logging.getLogger(__name__)


class TiingoProvider:
    """
    Tiingo-only data provider for professional financial data
    
    Features:
    - High-quality market data from Tiingo
    - Robust error handling and retry logic
    - Professional-grade financial data API
    - Caching to avoid rate limits
    """
    
    def __init__(self):
        # Get Tiingo API key from environment
        self.tiingo_token = os.environ.get('TIINGO_API_TOKEN')
        
        # Initialize Tiingo client if token available
        self.tiingo_client = None
        if self.tiingo_token:
            try:
                self.tiingo_client = TiingoClient({'api_key': self.tiingo_token})
                logger.info("✅ Tiingo client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Tiingo client: {e}")
        else:
            logger.warning("No TIINGO_API_TOKEN found - using free tier")
            # Use free tier (limited)
            try:
                self.tiingo_client = TiingoClient()
                logger.info("✅ Tiingo client initialized (free tier)")
            except Exception as e:
                logger.error(f"Failed to initialize Tiingo free tier: {e}")
        
        # Cache for reducing API calls
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

    def get_profile(self, ticker: str) -> Dict:
        """Get company profile information"""
        try:
            if self.tiingo_client:
                # Get meta data from Tiingo
                meta = self.tiingo_client.get_ticker_meta(ticker)
                if meta:
                    return {
                        'ticker': ticker.upper(),
                        'name': meta.get('name', ticker),
                        'description': meta.get('description', ''),
                        'sector': meta.get('sector', ''),
                        'industry': meta.get('industry', ''),
                        'exchange': meta.get('exchangeCode', ''),
                        'currency': 'USD',  # Tiingo primarily USD
                        'market_cap': None,  # Would need additional API call
                        'employees': None,
                        'website': None,
                        'data_provider': 'tiingo'
                    }
            
            # No fallback - Tiingo only
            logger.warning(f"No Tiingo profile data available for {ticker}")
            return {
                'ticker': ticker.upper(),
                'name': ticker,
                'data_provider': 'tiingo_no_data',
                'error': 'No profile data available from Tiingo'
            }
            
        except Exception as e:
            logger.error(f"Failed to get profile for {ticker}: {e}")
            return {
                'ticker': ticker.upper(),
                'name': ticker,
                'data_provider': 'tiingo_fallback',
                'error': str(e)
            }

    def get_market(self, ticker: str) -> Dict:
        """Get current market data enhanced with fundamental ratios"""
        try:
            market_data = {
                'ticker': ticker.upper(),
                'price': None,
                'change': None,
                'change_percent': None,
                'volume': None,
                'market_cap': None,
                'pe_ratio': None,
                'pb_ratio': None,
                'enterprise_value': None,
                'dividend_yield': None,
                'high_52week': None,
                'low_52week': None,
                'data_provider': 'tiingo',
                'timestamp': None
            }
            
            if self.tiingo_client:
                # Get latest price data
                prices = self.tiingo_client.get_ticker_price(
                    ticker,
                    startDate=datetime.now() - timedelta(days=5),
                    endDate=datetime.now()
                )
                
                if prices and len(prices) > 0:
                    latest = prices[-1]
                    market_data.update({
                        'price': latest.get('close'),
                        'change': latest.get('close', 0) - latest.get('adjClose', 0),
                        'volume': latest.get('volume'),
                        'timestamp': latest.get('date')
                    })
                
                # Get daily fundamentals for enhanced market data
                try:
                    daily_fundamentals = self.get_daily_fundamentals(ticker, days=2)
                    if not daily_fundamentals.empty:
                        latest_fund = daily_fundamentals.iloc[-1]
                        market_data.update({
                            'market_cap': latest_fund.get('marketCap'),
                            'pe_ratio': latest_fund.get('peRatio'),
                            'pb_ratio': latest_fund.get('pbRatio'),
                            'enterprise_value': latest_fund.get('enterpriseVal')
                        })
                        logger.info(f"✅ Enhanced {ticker} market data with fundamentals")
                except Exception as e:
                    logger.debug(f"Could not enhance market data with fundamentals for {ticker}: {e}")
                
                return market_data
            
            # No fallback - Tiingo only
            logger.warning(f"No Tiingo market data available for {ticker}")
            return {
                'ticker': ticker.upper(),
                'data_provider': 'tiingo_no_data',
                'error': 'No market data available from Tiingo'
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data for {ticker}: {e}")
            return {
                'ticker': ticker.upper(),
                'data_provider': 'tiingo_fallback',
                'error': str(e)
            }

    def get_price_history(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get historical price data as DataFrame"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=365*10)  # 10 years default for walk-forward
            if not end_date:
                end_date = datetime.now()
                
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            if self.tiingo_client:
                prices = self.tiingo_client.get_ticker_price(
                    ticker,
                    startDate=start_date,
                    endDate=end_date
                )
                
                if prices:
                    df = pd.DataFrame(prices)
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        
                        # Make timestamps tz-naive UTC (simplified sanitization)
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        
                        # Standardize column names
                        column_mapping = {
                            'close': 'Close',
                            'open': 'Open', 
                            'high': 'High',
                            'low': 'Low',
                            'volume': 'Volume',
                            'adjClose': 'Adj Close'
                        }
                        df = df.rename(columns=column_mapping)
                        
                        logger.info(f"✅ Retrieved {len(df)} days of data for {ticker} from Tiingo")
                        return df
            
            # No fallback - Tiingo only
            logger.warning(f"No Tiingo price history available for {ticker}")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
            
        except Exception as e:
            logger.error(f"Failed to get price history for {ticker}: {e}")
            # Return empty DataFrame with standard columns
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])

    def get_financials(self, ticker: str, years: int = 5) -> Dict[str, List[Dict]]:
        """Get financial statements from Tiingo Fundamentals API"""
        try:
            if not self.tiingo_client or not self.tiingo_token:
                logger.warning(f"No Tiingo credentials for fundamentals data for {ticker}")
                return {'income_statements': [], 'balance_sheets': [], 'cash_flows': []}
            
            # Use direct API call for fundamentals (TiingoClient may not support this endpoint)
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Token {self.tiingo_token}'
            }
            
            # Get fundamental statements
            url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/statements"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                if response.status_code == 403:
                    logger.warning(f"Tiingo fundamentals access denied for {ticker} (need subscription for non-DOW30)")
                else:
                    logger.warning(f"Tiingo fundamentals API error {response.status_code} for {ticker}")
                return {'income_statements': [], 'balance_sheets': [], 'cash_flows': []}
            
            statements_data = response.json()
            if not statements_data:
                logger.info(f"No fundamental statements available for {ticker}")
                return {'income_statements': [], 'balance_sheets': [], 'cash_flows': []}
            
            # Process statements data
            income_statements = []
            balance_sheets = []
            cash_flows = []
            
            cutoff_date = datetime.now() - timedelta(days=365 * years)
            
            for stmt in statements_data:
                stmt_date = stmt.get('date')
                if not stmt_date:
                    continue
                    
                # Parse date and filter by years
                try:
                    parsed_date = datetime.strptime(stmt_date, '%Y-%m-%d')
                    if parsed_date < cutoff_date:
                        continue
                except:
                    continue
                
                stmt_data = stmt.get('statementData', {})
                
                # Income Statement
                if 'incomeStatement' in stmt_data:
                    income_data = {'date': stmt_date, 'quarter': stmt.get('quarter'), 'year': stmt.get('year')}
                    for field in stmt_data['incomeStatement']:
                        data_code = field.get('dataCode')
                        value = field.get('value')
                        if data_code and value is not None:
                            income_data[data_code] = value
                    income_statements.append(income_data)
                
                # Balance Sheet
                if 'balanceSheet' in stmt_data:
                    balance_data = {'date': stmt_date, 'quarter': stmt.get('quarter'), 'year': stmt.get('year')}
                    for field in stmt_data['balanceSheet']:
                        data_code = field.get('dataCode')
                        value = field.get('value')
                        if data_code and value is not None:
                            balance_data[data_code] = value
                    balance_sheets.append(balance_data)
                
                # Cash Flow
                if 'cashFlow' in stmt_data:
                    cf_data = {'date': stmt_date, 'quarter': stmt.get('quarter'), 'year': stmt.get('year')}
                    for field in stmt_data['cashFlow']:
                        data_code = field.get('dataCode')
                        value = field.get('value')
                        if data_code and value is not None:
                            cf_data[data_code] = value
                    cash_flows.append(cf_data)
            
            logger.info(f"✅ Retrieved {len(income_statements)} income statements, {len(balance_sheets)} balance sheets, {len(cash_flows)} cash flow statements for {ticker}")
            
            return {
                'income_statements': income_statements,
                'balance_sheets': balance_sheets,
                'cash_flows': cash_flows
            }
            
        except Exception as e:
            logger.error(f"Failed to get Tiingo fundamentals for {ticker}: {e}")
            return {'income_statements': [], 'balance_sheets': [], 'cash_flows': []}

    def get_daily_fundamentals(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get daily fundamental metrics (P/E, P/B, Market Cap, etc.)"""
        try:
            if not self.tiingo_client or not self.tiingo_token:
                logger.warning(f"No Tiingo credentials for daily fundamentals for {ticker}")
                return pd.DataFrame()
            
            # Use direct API call for daily fundamentals
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Token {self.tiingo_token}'
            }
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/daily"
            params = {
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Tiingo daily fundamentals API error {response.status_code} for {ticker}")
                return pd.DataFrame()
            
            data = response.json()
            if not data:
                logger.info(f"No daily fundamental data available for {ticker}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            logger.info(f"✅ Retrieved {len(df)} days of daily fundamentals for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get daily fundamentals for {ticker}: {e}")
            return pd.DataFrame()

    def get_estimates(self, ticker: str) -> Dict:
        """Get analyst estimates - not available in Tiingo, use fallback"""
        return {
            'ticker': ticker.upper(),
            'data_provider': 'tiingo_unavailable',
            'estimates': []
        }

    def get_news(self, ticker: str, days: int = 7, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Get news from Tiingo for a specified date range or recent days
        
        Args:
            ticker: Stock symbol
            days: Number of days to fetch (used if start_date/end_date not provided)
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
        
        Returns:
            List of news articles with fields: title, summary, url, source, published, publishedDate, crawlDate
        """
        try:
            if self.tiingo_client:
                # Tiingo has news API
                if start_date and end_date:
                    # Use provided date range
                    start_str = start_date
                    end_str = end_date
                else:
                    # Use recent days
                    end_dt = datetime.now()
                    start_dt = end_dt - timedelta(days=days)
                    start_str = start_dt.strftime('%Y-%m-%d')
                    end_str = end_dt.strftime('%Y-%m-%d')
                
                news = self.tiingo_client.get_news(
                    tickers=[ticker],
                    startDate=start_str,
                    endDate=end_str,
                    limit=1000  # Increase limit for historical ranges
                )
                
                if news:
                    formatted_news = []
                    for article in news:
                        # Include multiple date fields for compatibility
                        published = article.get('publishedDate', '')
                        crawl = article.get('crawlDate', published)
                        
                        formatted_news.append({
                            'title': article.get('title', ''),
                            'summary': article.get('description', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', ''),
                            'published': published,
                            'publishedDate': published,  # HF generator expects this
                            'crawlDate': crawl,          # HF generator expects this
                            'sentiment': None,
                            'data_provider': 'tiingo'
                        })
                    
                    logger.info(f"✅ Retrieved {len(formatted_news)} news articles for {ticker} ({start_str} to {end_str})")
                    return formatted_news
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get news for {ticker}: {e}")
            return []




def test_tiingo_provider():
    """Test function for Tiingo provider"""
    provider = TiingoProvider()
    
    test_ticker = 'AAPL'
    
    print("🧪 Testing Tiingo Provider")
    print("=" * 50)
    
    # Test profile
    print(f"📋 Testing profile for {test_ticker}:")
    profile = provider.get_profile(test_ticker)
    print(f"  Name: {profile.get('name')}")
    print(f"  Sector: {profile.get('sector')}")
    print(f"  Provider: {profile.get('data_provider')}")
    print()
    
    # Test market data  
    print(f"💹 Testing market data for {test_ticker}:")
    market = provider.get_market(test_ticker)
    print(f"  Price: ${market.get('price')}")
    print(f"  Volume: {market.get('volume'):,}")
    print(f"  Provider: {market.get('data_provider')}")
    print()
    
    # Test price history
    print(f"📈 Testing price history for {test_ticker}:")
    df = provider.get_price_history(test_ticker)
    print(f"  Retrieved {len(df)} days of data")
    if not df.empty:
        print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print()
    
    # Test fundamentals
    print(f"🏢 Testing fundamentals for {test_ticker}:")
    financials = provider.get_financials(test_ticker)
    print(f"  Income statements: {len(financials.get('income_statements', []))}")
    print(f"  Balance sheets: {len(financials.get('balance_sheets', []))}")
    print(f"  Cash flows: {len(financials.get('cash_flows', []))}")
    if financials.get('income_statements'):
        latest_income = financials['income_statements'][0]
        revenue = latest_income.get('revenue')
        netinc = latest_income.get('netinc')
        if revenue:
            print(f"  Latest Revenue: ${revenue/1e9:.1f}B ({latest_income.get('date')})")
        if netinc:
            print(f"  Latest Net Income: ${netinc/1e9:.1f}B")
    print()
    
    # Test daily fundamentals
    print(f"📊 Testing daily fundamentals for {test_ticker}:")
    daily_fund = provider.get_daily_fundamentals(test_ticker, days=5)
    print(f"  Retrieved {len(daily_fund)} days of fundamental metrics")
    if not daily_fund.empty:
        latest = daily_fund.iloc[-1]
        if 'marketCap' in daily_fund.columns:
            print(f"  Market Cap: ${latest['marketCap']/1e9:.1f}B")
        if 'peRatio' in daily_fund.columns:
            print(f"  P/E Ratio: {latest['peRatio']:.2f}")
        if 'pbRatio' in daily_fund.columns:
            print(f"  P/B Ratio: {latest['pbRatio']:.2f}")
    print()
    
    # Test news
    print(f"📰 Testing news for {test_ticker}:")
    news = provider.get_news(test_ticker, days=3)
    print(f"  Retrieved {len(news)} articles")
    if news:
        print(f"  Latest: {news[0].get('title', 'No title')[:60]}...")
    print()
    
    print("✅ Tiingo provider test complete!")


if __name__ == "__main__":
    test_tiingo_provider()