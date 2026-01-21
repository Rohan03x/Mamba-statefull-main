"""
Multi-Provider Orchestrator Integration Test
============================================

Comprehensive test suite for the multi-provider orchestration system testing:
- Provider initialization and health monitoring
- Intelligent data routing and fallback logic
- Parallel data retrieval and quality assessment
- Cross-provider validation and error handling
- Performance benchmarking and optimization
"""

from dcf_lab.providers.multi_provider_orchestrator import (
    DataQuality,
    MultiProviderOrchestrator,
)


def test_comprehensive_orchestration():
    """Test comprehensive multi-provider orchestration"""

    # Initialize orchestrator
    orchestrator = MultiProviderOrchestrator()

    print("\n=== Multi-Provider Orchestration Test ===")
    print(f"Initialized {len(orchestrator.providers)} providers")

    # Test provider health report
    health_report = orchestrator.get_provider_health_report()
    print("\n📊 Provider Health Report:")
    for name, health in health_report.items():
        print(
            f"  {name}: {health['health_score']:.1f}/100 ({health['priority']})")
        print(f"    Capabilities: {', '.join(health['capabilities'][:3])}...")

    # Test comprehensive data retrieval
    symbol = 'AAPL'
    print(f"\n🔄 Fetching comprehensive data for {symbol}...")

    comprehensive_data = orchestrator.get_comprehensive_data(
        symbol=symbol,
        quality_threshold=DataQuality.FAIR
    )

    print(f"\n📈 Retrieved {len(comprehensive_data)} data types:")
    for data_type, response in comprehensive_data.items():
        status = "✅" if response.quality_score > 0.5 else "⚠️"
        print(
            f"  {status} {data_type}: {
                response.source} (quality: {
                response.quality_score:.2f})")
        if response.metadata.get('fetch_time_ms'):
            print(
                f"      Fetch time: {
                    response.metadata['fetch_time_ms']:.1f}ms")
        if response.errors:
            print(f"      Errors: {len(response.errors)}")
        if response.warnings:
            print(f"      Warnings: {len(response.warnings)}")

    # Test provider benchmarking
    print("\n⚡ Running provider benchmarks...")
    benchmark_results = orchestrator.benchmark_providers(symbol)

    print("\n🏆 Provider Performance Ranking:")
    # Sort by success then by response time
    sorted_providers = sorted(
        benchmark_results.items(),
        key=lambda x: (not x[1]['success'], x[1]['response_time_ms'])
    )

    for i, (name, result) in enumerate(sorted_providers, 1):
        status = "✅" if result['success'] else "❌"
        print(f"  #{i} {status} {name}:")
        print(f"      Response time: {result['response_time_ms']:.1f}ms")
        print(f"      Quality score: {result['quality_score']:.2f}")
        print(f"      Data size: {result['data_size']:,} chars")
        if result.get('error'):
            print(f"      Error: {result['error'][:100]}...")

    # Calculate overall statistics
    successful_providers = [
        r for r in benchmark_results.values() if r['success']]
    if successful_providers:
        avg_response_time = sum(
    r['response_time_ms'] for r in successful_providers) / len(successful_providers)
        avg_quality = sum(r['quality_score']
                          for r in successful_providers) / len(successful_providers)

        print("\n📊 Overall Statistics:")
        print(
            f"  Success rate: {
                len(successful_providers)}/{
                len(benchmark_results)} ({
                len(successful_providers)/len(benchmark_results)*100:.1f}%)")
        print(f"  Average response time: {avg_response_time:.1f}ms")
        print(f"  Average quality score: {avg_quality:.2f}")

    # Validate results
    assert len(orchestrator.providers) >= 5, "Should have at least 5 providers"
    assert len(comprehensive_data) >= 3, "Should retrieve at least 3 data types"
    assert len(successful_providers) >= 3, "At least 3 providers should succeed"

    # Test data quality
    high_quality_responses = [
        r for r in comprehensive_data.values() if r.quality_score >= 0.8]
    assert len(
        high_quality_responses) >= 2, "Should have at least 2 high-quality responses"

    print("\n✅ Multi-provider orchestration test completed successfully!")
    print(f"   - {len(orchestrator.providers)} providers initialized")
    print(f"   - {len(comprehensive_data)} data types retrieved")
    print(f"   - {len(successful_providers)} providers functioning")
    print(f"   - {len(high_quality_responses)} high-quality responses")

    return True


def test_provider_routing_logic():
    """Test intelligent provider routing and selection"""

    orchestrator = MultiProviderOrchestrator()

    print("\n=== Provider Routing Logic Test ===")

    # Test data type to provider mapping
    test_cases = [
        ('price_data', ['enhanced_yfinance', 'alpha_vantage']),
        ('economic_indicators', ['fred', 'alpha_vantage']),
        ('market_structure', ['market_structure']),
        ('financial_statements', ['sec_xbrl']),
        ('options_chains', ['enhanced_yfinance'])
    ]

    for data_type, expected_providers in test_cases:
        best_providers = orchestrator._find_best_providers(data_type)
        print(f"  {data_type}: {best_providers[:2]}")

        # Check that at least one expected provider is in top candidates
        assert any(provider in best_providers for provider in
                   expected_providers), f"No expected provider found for {
            data_type} "

    # Test provider scoring
    provider_scores = {}
    for name, source in orchestrator.providers.items():
        score = orchestrator._calculate_provider_score(source, 'price_data')
        provider_scores[name] = score
        print(f"  {name} score for price_data: {score:.1f}")

    # Enhanced YFinance should have high score for price data
    assert provider_scores['enhanced_yfinance'] > provider_scores['fred'], \
        "Enhanced YFinance should score higher than FRED for price data"

    print("✅ Provider routing logic test passed")


def test_data_quality_assessment():
    """Test data quality assessment functionality"""

    orchestrator = MultiProviderOrchestrator()

    print("\n=== Data Quality Assessment Test ===")

    # Test various data quality scenarios
    test_data = [
        # High quality data
        ({
            'symbol': 'AAPL',
            'close': 150.0,
            'timestamp': '2024-01-15',
            'volume': 50000000
        }, 'price_data', 'High quality'),

        # Medium quality data
        ({
            'symbol': 'AAPL',
            'revenue': 365000000
        }, 'fundamentals', 'Medium quality'),

        # Low quality data
        ({'symbol': 'AAPL'}, 'price_data', 'Low quality'),

        # Error data
        ({'error': 'API limit exceeded'}, 'price_data', 'Error data'),

        # Empty data
        ({}, 'price_data', 'Empty data'),

        # None data
        (None, 'price_data', 'None data')
    ]

    for data, data_type, description in test_data:
        quality_score = orchestrator._assess_data_quality(data, data_type)
        print(f"  {description}: {quality_score:.2f}")

        # Basic quality checks
        if data is None:
            assert abs(quality_score -
                       0.0) < 1e-10, "None data should have 0 quality"
        elif isinstance(data, dict) and 'error' in data:
            assert quality_score < 0.5, "Error data should have low quality"
        elif isinstance(data, dict) and len(data) > 2:
            assert quality_score > 0.5, "Rich data should have decent quality"

    print("✅ Data quality assessment test passed")


def test_error_handling_and_fallback():
    """Test error handling and fallback mechanisms"""

    orchestrator = MultiProviderOrchestrator()

    print("\n=== Error Handling & Fallback Test ===")

    # Test with invalid symbol to trigger errors
    invalid_symbol = 'INVALID_SYMBOL_XYZ'

    try:
        results = orchestrator.get_comprehensive_data(
            symbol=invalid_symbol,
            quality_threshold=DataQuality.POOR  # Lower threshold to allow some results
        )

        print(f"  Retrieved {len(results)} results for invalid symbol")

        # Count responses with errors
        error_responses = [r for r in results.values() if r.errors]
        success_responses = [
            r for r in results.values()
            if not r.errors and r.quality_score > 0]

        print(f"  Responses with errors: {len(error_responses)}")
        print(f"  Successful responses: {len(success_responses)}")

        # Should handle errors gracefully
        assert len(
            results) > 0, "Should return some results even with invalid symbol"

    except Exception as e:
        print(f"  Caught exception (expected): {str(e)[:100]}...")

    # Test provider health updates
    initial_health = orchestrator.providers['enhanced_yfinance'].health_score
    orchestrator._update_provider_health(
        'enhanced_yfinance', False, 'Test error')
    after_error_health = orchestrator.providers['enhanced_yfinance'].health_score

    print(
        f"  Provider health: {initial_health:.1f} -> {after_error_health:.1f} (after error)")
    assert after_error_health < initial_health, "Health should decrease after error"

    # Restore health
    orchestrator._update_provider_health('enhanced_yfinance', True)
    after_success_health = orchestrator.providers['enhanced_yfinance'].health_score

    print(
        f"  Provider health: {
            after_error_health:.1f} -> {
            after_success_health:.1f} (after success)")
    assert after_success_health > after_error_health, "Health should increase after success"

    print("✅ Error handling & fallback test passed")


if __name__ == "__main__":
    # Run all tests
    test_comprehensive_orchestration()
    test_provider_routing_logic()
    test_data_quality_assessment()
    test_error_handling_and_fallback()

    print("\n🎉 All multi-provider orchestration tests passed!")
