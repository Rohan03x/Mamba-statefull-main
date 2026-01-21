"""
Market Structure Provider Integration Tests
==========================================

Test suite for market microstructure data provider covering:
- Short interest data retrieval and analysis
- Dark pool trading metrics
- Options flow and unusual activity detection
- Market maker order flow analysis
- Trading venue statistics
- Comprehensive market structure scoring
"""

from datetime import datetime

from dcf_lab.providers.market_structure_provider import (
    DarkPoolData,
    MarketStructureProvider,
    ShortInterestData,
)


class TestMarketStructureProvider:
    """Test suite for MarketStructureProvider"""

    def setup_method(self):
        """Set up test fixtures"""
        self.provider = MarketStructureProvider()
        self.test_symbol = 'AAPL'

    def test_short_interest_data_retrieval(self):
        """Test short interest data retrieval"""
        # Test with default date range
        short_data = self.provider.get_short_interest_data(self.test_symbol)

        assert isinstance(short_data, list)
        assert len(short_data) > 0

        # Check data structure
        for data_point in short_data:
            assert isinstance(data_point, ShortInterestData)
            assert data_point.symbol == self.test_symbol
            assert data_point.short_interest >= 0
            assert data_point.avg_daily_volume > 0
            assert data_point.days_to_cover >= 0

    def test_short_interest_custom_date_range(self):
        """Test short interest with custom date range"""
        start_date = '2024-01-01'
        end_date = '2024-03-31'

        short_data = self.provider.get_short_interest_data(
            self.test_symbol, start_date, end_date
        )

        assert len(short_data) >= 6  # Bi-monthly reporting for 3 months

        # Verify date range
        dates = [
            datetime.strptime(
                d.settlement_date,
                '%Y-%m-%d') for d in short_data]
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        assert all(start_dt <= date <= end_dt for date in dates)

    def test_dark_pool_data_retrieval(self):
        """Test dark pool data retrieval"""
        dark_pool_data = self.provider.get_dark_pool_data(self.test_symbol)

        assert isinstance(dark_pool_data, DarkPoolData)
        assert dark_pool_data.symbol == self.test_symbol
        assert dark_pool_data.dark_pool_volume > 0
        assert dark_pool_data.total_volume > dark_pool_data.dark_pool_volume
        assert 0 < dark_pool_data.dark_pool_percentage < 100
        assert isinstance(dark_pool_data.venue_breakdown, dict)
        assert len(dark_pool_data.venue_breakdown) > 0

        # Verify venue breakdown sums to total dark pool volume
        total_venue_volume = sum(dark_pool_data.venue_breakdown.values())
        assert abs(
            total_venue_volume -
            dark_pool_data.dark_pool_volume) < 1000  # Allow small rounding

    def test_options_flow_data(self):
        """Test options flow data retrieval"""
        options_data = self.provider.get_options_flow_data(self.test_symbol)

        assert isinstance(options_data, dict)
        assert options_data['symbol'] == self.test_symbol
        assert options_data['total_volume'] > 0
        assert options_data['call_volume'] + \
            options_data['put_volume'] == options_data['total_volume']
        assert options_data['call_put_ratio'] > 0
        assert isinstance(options_data['unusual_activity'], list)

    def test_market_maker_data(self):
        """Test market maker data retrieval"""
        mm_data = self.provider.get_market_maker_data(self.test_symbol)

        assert isinstance(mm_data, dict)
        assert mm_data['symbol'] == self.test_symbol
        assert mm_data['total_market_maker_volume'] > 0
        assert 0 < mm_data['market_maker_percentage'] < 1
        assert isinstance(mm_data['top_market_makers'], list)
        assert len(mm_data['top_market_makers']) > 0

        # Check order flow metrics
        assert 'order_flow_metrics' in mm_data
        assert 'payment_for_order_flow' in mm_data['order_flow_metrics']
        assert 'internalization_rate' in mm_data['order_flow_metrics']

        # Check liquidity metrics
        assert 'liquidity_metrics' in mm_data
        assert 'bid_ask_spread_bps' in mm_data['liquidity_metrics']
        assert 'market_depth_shares' in mm_data['liquidity_metrics']

        # Verify market maker shares sum to reasonable total
        total_share = sum(mm['market_share']
                          for mm in mm_data['top_market_makers'])
        assert 90 <= total_share <= 110  # Allow for rounding

    def test_venue_statistics(self):
        """Test trading venue statistics"""
        venue_data = self.provider.get_venue_statistics(self.test_symbol)

        assert isinstance(venue_data, dict)
        assert venue_data['symbol'] == self.test_symbol
        assert venue_data['total_volume'] > 0
        assert isinstance(venue_data['venue_breakdown'], dict)

        # Check venue breakdown
        total_percentage = sum(v['percentage']
                               for v in venue_data['venue_breakdown'].values())
        assert 99 <= total_percentage <= 101  # Allow for rounding

        # Check metrics
        assert 'metrics' in venue_data
        assert 'fragmentation_index' in venue_data['metrics']
        assert 'primary_venue_share' in venue_data['metrics']
        assert 0 <= venue_data['metrics']['fragmentation_index'] <= 1

    def test_comprehensive_market_structure(self):
        """Test comprehensive market structure analysis"""
        comprehensive_data = self.provider.get_comprehensive_market_structure(
            self.test_symbol)

        assert isinstance(comprehensive_data, dict)
        assert comprehensive_data['symbol'] == self.test_symbol
        assert 'timestamp' in comprehensive_data
        assert isinstance(comprehensive_data['data_sources'], list)
        assert len(comprehensive_data['data_sources']) > 0

        # Check for expected data sections
        expected_sections = [
            'short_interest',
            'dark_pool',
            'options_flow',
            'market_makers',
            'venue_statistics']
        present_sections = [
            section for section in expected_sections
            if section in comprehensive_data]
        # At least 3 sections should be present
        assert len(present_sections) >= 3

        # Check market structure score
        assert 'market_structure_score' in comprehensive_data
        score_data = comprehensive_data['market_structure_score']
        assert 'score' in score_data
        assert 'rating' in score_data
        assert 0 <= score_data['score'] <= 100
        assert score_data['rating'] in ['poor', 'fair', 'good', 'excellent']

    def test_rate_limiting(self):
        """Test rate limiting functionality"""

        # Make multiple requests to same source
        self.provider._rate_limit('finra')
        first_call = datetime.now()

        self.provider._rate_limit('finra')
        second_call = datetime.now()

        # Should have at least the minimum interval between calls
        elapsed = (second_call - first_call).total_seconds()
        min_interval = self.provider._min_request_intervals['finra']
        assert elapsed >= min_interval - 0.1  # Allow small timing variance

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with invalid symbol
        short_data = self.provider.get_short_interest_data('')
        assert isinstance(short_data, list)
        # Test should handle gracefully - length is always >= 0, no need to
        # assert

        # Test with invalid date format
        short_data = self.provider.get_short_interest_data(
            self.test_symbol, 'invalid-date')
        assert isinstance(short_data, list)  # Should handle gracefully

    def test_data_consistency(self):
        """Test data consistency across multiple calls"""
        # Get comprehensive data twice
        data1 = self.provider.get_comprehensive_market_structure(
            self.test_symbol)
        data2 = self.provider.get_comprehensive_market_structure(
            self.test_symbol)

        # Structure should be consistent
        assert data1['symbol'] == data2['symbol']
        assert set(data1.keys()) == set(data2.keys())

        # Data sources should be consistent
        assert set(data1['data_sources']) == set(data2['data_sources'])

    def test_short_interest_trend_analysis(self):
        """Test short interest trend analysis"""
        # Get historical data
        short_data = self.provider.get_short_interest_data(self.test_symbol)

        if len(short_data) >= 2:
            trend_analysis = self.provider._analyze_short_interest_trend(
                short_data)

            assert isinstance(trend_analysis, dict)
            assert 'trend' in trend_analysis
            assert trend_analysis['trend'] in [
                'increasing', 'decreasing', 'stable', 'insufficient_data']

            if trend_analysis['trend'] != 'insufficient_data':
                assert 'latest_days_to_cover' in trend_analysis
                assert 'avg_days_to_cover' in trend_analysis
                assert 'volatility' in trend_analysis

    def test_structure_score_calculation(self):
        """Test market structure score calculation"""
        # Create mock comprehensive data
        mock_data = {
            'short_interest': {
                'latest': {'days_to_cover': 2.5}  # Low short interest (good)
            },
            'dark_pool': {
                'dark_pool_percentage': 20.0  # Healthy level
            },
            'options_flow': {
                'call_put_ratio': 1.2  # Balanced
            },
            'market_makers': {
                'market_maker_percentage': 0.35  # Healthy level
            },
            'venue_statistics': {
                # Appropriate fragmentation
                'metrics': {'fragmentation_index': 0.7}
            }
        }

        score_data = self.provider._calculate_structure_score(mock_data)

        assert isinstance(score_data, dict)
        assert 'score' in score_data
        assert 'rating' in score_data
        assert 'contributing_factors' in score_data
        assert 0 <= score_data['score'] <= 100

        # Should have good score with healthy metrics
        assert score_data['score'] > 60
        assert len(score_data['contributing_factors']) > 0

# Integration test function


def test_market_structure_integration():
    """Integration test for market structure provider"""
    provider = MarketStructureProvider()

    # Test with real-world symbol
    result = provider.get_comprehensive_market_structure('AAPL')

    print("\n=== Market Structure Analysis for AAPL ===")
    print(f"Data Sources: {result.get('data_sources', [])}")

    if 'short_interest' in result:
        si = result['short_interest']['latest']
        print(
            f"Short Interest: {
                si['short_interest']:,                } shares ({
                si['days_to_cover']:.1f} days to cover)")

    if 'dark_pool' in result:
        dp = result['dark_pool']
        print(
            f"Dark Pool Activity: {
                dp['dark_pool_percentage']:.1f}% of volume")

    if 'market_structure_score' in result:
        score = result['market_structure_score']
        print(f"Structure Score: {score['score']}/100 ({score['rating']})")

    assert 'symbol' in result
    assert result['symbol'] == 'AAPL'
    print("✅ Market structure integration test passed")


if __name__ == "__main__":
    test_market_structure_integration()
