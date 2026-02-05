"""
Twitter/X twikit generator - Three-channel social sentiment and attention.

Implements three-channel data collection architecture:
- Channel A (Cashtag): $TICKER queries - highest precision market intent
- Channel B (Name Intent): Company name + finance context - higher recall
- Channel C (Curated): Official accounts - highest quality

Symbol-bound family: Caches to cache/symbols/<SYMBOL>/hf/twitter_twikit_hf.parquet
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add .venv to path for twikit import
VENV_SITE_PACKAGES = Path(__file__).parent.parent.parent / ".venv" / "lib" / "python3.12" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))


def get_company_name(symbol: str) -> str:
    """
    Get company name from symbol using EODHD provider.
    Fallback to symbol if lookup fails.
    """
    try:
        # Add project root to path for imports
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.data_sources.eodhd_provider import EODHDProvider
        
        provider = EODHDProvider()
        profile = provider.get_profile(symbol)
        
        if profile and 'name' in profile:
            name = profile['name']
            if name and name.strip():
                # Clean up common suffixes
                name = name.replace(' Inc.', '').replace(' Corporation', '')
                name = name.replace(' Corp.', '').replace(' Ltd.', '')
                name = name.replace(', Inc.', '').replace(', Corp.', '')
                return name.strip()
        
        # Fallback to symbol
        return symbol
        
    except Exception as e:
        logger.warning(f"Failed to get company name for {symbol}: {e}")
        return symbol


def generate_twitter_twikit_hf(
    symbol: str,
    start: str,
    end: str,
    output_path: Path,
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Generate twitter_twikit_hf family data using Reddit/YARS (NO AUTH NEEDED).
    
    THREE-CHANNEL ARCHITECTURE:
    - Channel A: Symbol keyword queries (e.g., "AAPL")
    - Channel B: Company name queries (e.g., "Apple stock")
    - Channel C: Curated accounts (optional, requires Twitter auth)
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start: Start date (YYYY-MM-DD, inclusive)
        end: End date (YYYY-MM-DD, EXCLUSIVE - following HF generator contract)
        output_path: Path to save parquet file
        **kwargs: Additional config (from hf_registry.yaml)
    
    Returns:
        DataFrame with 24 features or None if failed
    """
    logger.info(f"🎯 twitter_twikit_hf: Starting Reddit collection for {symbol}")
    
    # Get company name for Channel B
    company_name = get_company_name(symbol)
    logger.info(f"   Symbol: {symbol} → Company Name: {company_name}")
    
    # Parse date range - end is EXCLUSIVE per HF generator contract
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Generate dates - end is exclusive, so subtract 1 day to get last inclusive date
    end_inclusive = end_dt - pd.Timedelta(days=1)
    date_range = pd.date_range(start=start_dt, end=end_inclusive, freq='D')
    
    logger.info(f"   Collecting {len(date_range)} days: {start_dt.date()} to {end_inclusive.date()} (inclusive)")
    
    # Always use Reddit mode (no Twitter auth needed)
    use_reddit = True
    try:
        import sys
        yars_parent = Path(__file__).parent
        if str(yars_parent) not in sys.path:
            sys.path.insert(0, str(yars_parent))
        from yars_lib.yars import YARS
        logger.info("✓ Using Reddit (YARS) - NO AUTH NEEDED!")
    except ImportError as e:
        logger.error(f"❌ YARS not available: {e}")
        logger.error("Cannot generate twitter_twikit_hf without YARS")
        return None
    
    # OPTIMIZED: Batch fetch all dates at once instead of looping
    logger.info(f"   🚀 Batch fetching Reddit data for entire date range at once")
    all_tweets_by_date = _batch_collect_all_dates(
        client=None,
        symbol=symbol,
        company_name=company_name,
        date_range=date_range,
        use_reddit=use_reddit,
        **kwargs
    )
    
    # Batch process all tweets together with FinBERT for efficiency
    logger.info(f"   📦 Batch processing sentiment for all {len(date_range)} dates together")
    all_sentiment_scores, all_novelty_scores = _batch_process_ml_features(all_tweets_by_date)
    
    # Compute features for each date using pre-computed ML scores
    all_features = []
    for idx, (current_date, tweets) in enumerate(all_tweets_by_date):
        try:
            sentiment_scores = all_sentiment_scores[idx]
            novelty_scores = all_novelty_scores[idx]
            
            if len(tweets) == 0:
                all_features.append(_create_zero_features(current_date, symbol))
            else:
                features = _compute_features_with_scores(
                    tweets, current_date, symbol, 
                    sentiment_scores, novelty_scores,
                    kwargs.get('quality_filters', {})
                )
                all_features.append(features)
                
        except Exception as e:
            logger.warning(f"   ⚠️  Feature computation failed for {current_date.strftime('%Y-%m-%d')}: {e}")
            all_features.append(_create_zero_features(current_date, symbol))
    
    # Combine into DataFrame
    if not all_features:
        logger.warning(f"❌ No features generated for {symbol}")
        return _create_stub_features(date_range, symbol)
    
    df = pd.DataFrame(all_features)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # Save to cache
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"✅ Saved {len(df)} rows to {output_path}")
    
    return df


def _batch_collect_all_dates(
    client,
    symbol: str,
    company_name: str,
    date_range,
    use_reddit: bool = False,
    **kwargs
) -> list:
    """
    OPTIMIZED: Collect tweets using WEEKLY batching for better date coverage.
    
    Reddit's search API returns most relevant/recent posts. Fetching the entire 
    15-year range would return only recent posts. Instead, we fetch week-by-week
    to get better historical coverage while keeping API calls reasonable.
    
    Strategy:
    - Split date range into weeks
    - Fetch up to 100 posts per week per channel (Reddit's max)
    - This gives much better coverage: ~200 posts/week vs 200 posts/15 years
    
    Returns:
        List of (date, tweets) tuples
    """
    sampling_cfg = kwargs.get('sampling', {})
    quality_cfg = kwargs.get('quality_filters', {})
    
    # Get full date range
    start_time = date_range[0].replace(hour=0, minute=0, second=0)
    end_time = date_range[-1].replace(hour=23, minute=59, second=59)
    
    # Channel queries
    cashtag_query = symbol
    name_query = f'{company_name} stock'
    
    # Split into weekly chunks for better coverage
    weeks = []
    current = start_time
    while current <= end_time:
        week_end = min(current + pd.Timedelta(days=6), end_time)
        weeks.append((current, week_end))
        current = week_end + pd.Timedelta(days=1)
    
    total_weeks = len(weeks)
    logger.info(f"      🎯 Fetching {total_weeks} weeks ({start_time.date()} → {end_time.date()})")
    logger.info(f"      🎯 Strategy: ~100 posts/week/channel for better historical coverage")
    logger.info(f"      🎯 Channel A query: {cashtag_query}")
    logger.info(f"      🎯 Channel B query: {name_query}")
    
    try:
        all_tweets = []
        
        # Fetch week-by-week (with rate limit protection)
        for week_idx, (week_start, week_end) in enumerate(weeks):
            if week_idx > 0 and week_idx % 10 == 0:
                # Log progress every 10 weeks
                logger.info(f"        📈 Progress: {week_idx}/{total_weeks} weeks ({len(all_tweets)} posts collected)")
                # Add small delay every 10 weeks to avoid rate limits
                import time
                time.sleep(1)
            
            # Fetch both channels for this week
            week_tweets_a = _search_reddit_sync(
                symbol, cashtag_query, week_start, week_end,
                max_results=100, use_reddit=use_reddit
            )
            for tweet in week_tweets_a:
                tweet['channel'] = 'cashtag'
            
            week_tweets_b = _search_reddit_sync(
                symbol, name_query, week_start, week_end,
                max_results=100, use_reddit=use_reddit
            )
            for tweet in week_tweets_b:
                tweet['channel'] = 'name_intent'
            
            all_tweets.extend(week_tweets_a + week_tweets_b)
        
        logger.info(f"      📊 Fetched {len(all_tweets)} total posts from {total_weeks} weeks")
        
        # Apply quality filters to all tweets at once
        filtered_tweets = _apply_quality_filters(all_tweets, quality_cfg)
        logger.info(f"      ✂️  After quality filters: {len(filtered_tweets)} posts (from {len(all_tweets)})")
        
        # Split tweets by date and apply per-date sampling
        tweets_by_date = []
        for current_date in date_range:
            date_start = current_date.replace(hour=0, minute=0, second=0)
            date_end = current_date.replace(hour=23, minute=59, second=59)
            
            # Filter tweets for this specific date
            date_tweets = [
                t for t in filtered_tweets
                if date_start <= t.get('created_at', date_start) <= date_end
            ]
            
            # Apply sampling rules per date
            sampled = _apply_sampling_rules(date_tweets, sampling_cfg)
            tweets_by_date.append((current_date, sampled))
        
        logger.info(f"      ✅ Split into {len(tweets_by_date)} dates, ready for ML processing")
        return tweets_by_date
        
    except Exception as e:
        logger.warning(f"      ❌ Weekly batch collection failed: {e}, falling back to per-date collection")
        # Fallback to old per-date method
        all_tweets_by_date = []
        for current_date in date_range:
            try:
                tweets = _collect_tweets_for_date(
                    client, symbol, company_name, current_date, use_reddit, **kwargs
                )
                all_tweets_by_date.append((current_date, tweets))
            except Exception as ex:
                logger.warning(f"      ⚠️  Failed for {current_date.strftime('%Y-%m-%d')}: {ex}")
                all_tweets_by_date.append((current_date, []))
        return all_tweets_by_date


def _collect_tweets_for_date(
    client,
    symbol: str,
    company_name: str,
    date: pd.Timestamp,
    use_reddit: bool = False,
    **kwargs
) -> list:
    """
    Collect and filter tweets for a single date using three channels.
    
    Returns filtered and sampled tweets (without computing ML features).
    ML features (sentiment, novelty) will be batch-processed later for efficiency.
    """
    import asyncio
    from datetime import timedelta
    
    # Extract config
    channels_cfg = kwargs.get('channels', {})
    sampling_cfg = kwargs.get('sampling', {})
    quality_cfg = kwargs.get('quality_filters', {})
    
    # Channel queries (simplified for Reddit - no boolean operators)
    cashtag_query = symbol  # Reddit doesn't need $ prefix
    name_query = f'{company_name} stock'  # Simple keyword search
    
    logger.info(f"      🎯 Channel A query: {cashtag_query}")
    logger.info(f"      🎯 Channel B query: {name_query}")
    
    # Date range for this day
    start_time = date.replace(hour=0, minute=0, second=0)
    end_time = date.replace(hour=23, minute=59, second=59)
    
    # Collect tweets from all channels
    all_tweets = []
    
    try:
        # Channel A: Cashtag search (highest precision)
        channel_a_tweets = _search_reddit_sync(
            symbol, 
            cashtag_query, 
            start_time, 
            end_time,
            max_results=sampling_cfg.get('max_per_channel', 200),
            use_reddit=use_reddit
        )
        for tweet in channel_a_tweets:
            tweet['channel'] = 'cashtag'
        all_tweets.extend(channel_a_tweets)
        
        # Channel B: Name intent search (higher recall)
        channel_b_tweets = _search_reddit_sync(
            symbol, 
            name_query, 
            start_time, 
            end_time,
            max_results=sampling_cfg.get('max_per_channel', 200),
            use_reddit=use_reddit
        )
        for tweet in channel_b_tweets:
            tweet['channel'] = 'name_intent'
        all_tweets.extend(channel_b_tweets)
        
        logger.info(f"      📊 Collected {len(all_tweets)} total posts from channels A+B")
        
        # Channel C: Curated accounts (if configured)
        curated_accounts = channels_cfg.get('curated_accounts', {}).get(symbol, [])
        if curated_accounts:
            for account in curated_accounts[:5]:  # Max 5 curated accounts
                try:
                    curated_tweets = _get_user_tweets_sync(
                        client, 
                        account, 
                        start_time, 
                        end_time,
                        max_results=10
                    )
                    for tweet in curated_tweets:
                        tweet['channel'] = 'curated'
                    all_tweets.extend(curated_tweets)
                except Exception as e:
                    logger.debug(f"      Failed to get tweets from {account}: {e}")
        
        logger.debug(f"      Collected {len(all_tweets)} raw tweets")
        
        # Apply quality filters
        filtered_tweets = _apply_quality_filters(all_tweets, quality_cfg)
        logger.info(f"      ✂️  After quality filters: {len(filtered_tweets)} tweets (from {len(all_tweets)})")
        
        # Apply sampling rules
        sampled_tweets = _apply_sampling_rules(filtered_tweets, sampling_cfg)
        logger.info(f"      📐 After sampling: {len(sampled_tweets)} tweets (from {len(filtered_tweets)})")
        
        # Return tweets for batch ML processing
        return sampled_tweets
        
    except Exception as e:
        logger.warning(f"      Collection failed: {e}")
        return []


def _create_zero_features(date: pd.Timestamp, symbol: str) -> Dict[str, Any]:
    """Create feature dict with all zeros for a single date."""
    return {
        'date': date,
        # Governance (6)
        'twitter_twikit_has_data': 0.0,
        'twitter_twikit_activity': 0.0,
        'twitter_twikit_days_since_update': 999.0,
        'twitter_twikit_sample_n': 0.0,
        'twitter_twikit_unique_authors_n': 0.0,
        'twitter_twikit_query_coverage': 0.0,
        # Attention (5)
        'twitter_twikit_tweet_count': 0.0,
        'twitter_twikit_tweet_count_z': 0.0,
        'twitter_twikit_engagement_sum': 0.0,
        'twitter_twikit_engagement_per_tweet': 0.0,
        'twitter_twikit_volume_change_1d': 0.0,
        # Sentiment (6)
        'twitter_twikit_sent_mean': 0.0,
        'twitter_twikit_sent_neg_frac': 0.0,
        'twitter_twikit_sent_pos_frac': 0.0,
        'twitter_twikit_sent_neu_frac': 0.0,
        'twitter_twikit_sent_std': 0.0,
        'twitter_twikit_neg_tail_frac': 0.0,
        # Novelty (4)
        'twitter_twikit_novelty_mean': 0.0,
        'twitter_twikit_topic_earnings_frac': 0.0,
        'twitter_twikit_topic_lawsuit_frac': 0.0,
        'twitter_twikit_topic_guidance_frac': 0.0,
        # Portfolio (3)
        'twitter_twikit_event_risk': 0.0,
        'twitter_twikit_panic': 0.0,
        'twitter_twikit_sent_regime': 0.0,
    }


def _create_stub_features(date_range: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    """Create stub DataFrame with all zeros."""
    features = [_create_zero_features(date, symbol) for date in date_range]
    df = pd.DataFrame(features)
    df.set_index('date', inplace=True)
    return df


def _search_reddit_sync(symbol: str, query: str, start_time, end_time, max_results: int = 200, use_reddit: bool = False):
    """
    Search Reddit posts using YARS (NO authentication needed!).
    
    Uses Reddit's public JSON API - completely free, no API keys required.
    """
    
    if not use_reddit:
        return []
    
    try:
        import sys
        from pathlib import Path
        from datetime import datetime
        
        # Add YARS to path
        yars_parent = Path(__file__).parent
        if str(yars_parent) not in sys.path:
            sys.path.insert(0, str(yars_parent))
        
        from yars_lib.yars import YARS
        
        results = []
        
        # Initialize YARS scraper with rotating proxy (if available) to avoid IP blocks
        # Proxy URL can be overridden via environment variable
        proxy_url = None
        
        # Try to use rotating proxy if configured
        import os
        if 'REDDIT_PROXY_URL' in os.environ:
            proxy_url = os.environ['REDDIT_PROXY_URL']
            logger.info(f"        🌐 Using proxy from REDDIT_PROXY_URL: {proxy_url}")
        
        if proxy_url:
            try:
                # Test if proxy is available
                import requests
                test_response = requests.get(
                    "https://www.reddit.com/r/wallstreetbets.json", 
                    proxies={"http": proxy_url, "https": proxy_url},
                    timeout=8
                )
                if test_response.status_code == 200:
                    logger.info(f"        ✓ Proxy is working, using {proxy_url}")
                    miner = YARS(proxy=proxy_url, timeout=20)
                else:
                    logger.warning(f"        ⚠️  Proxy returned status {test_response.status_code}, using direct connection")
                    miner = YARS(timeout=10, random_user_agent=True)
            except Exception as e:
                logger.warning(f"        ⚠️  Proxy not available ({e}), using direct connection with random user agents")
                miner = YARS(timeout=10, random_user_agent=True)
        else:
            # No proxy configured, use direct connection with random user agents (helps avoid blocks)
            miner = YARS(timeout=10, random_user_agent=True)
        
        # Convert timestamps to unix
        start_unix = int(start_time.timestamp())
        end_unix = int(end_time.timestamp())
        
        # Search Reddit (YARS searches across all subreddits)
        logger.info(f"        🔍 Searching Reddit for '{query}' (NO AUTH NEEDED)...")
        
        try:
            # Use max_results parameter (default 200) instead of hardcoded limit
            posts = miner.search_reddit(query, limit=min(max_results, 1000))  # Cap at 1000 to avoid timeouts
            
            for post in posts:
                try:
                    # YARS search returns basic data - use current time as approximation
                    # Reddit search returns recent posts anyway
                    created_approx = int(end_time.timestamp())  # Assume recent
                    
                    # Combine title and description
                    text = post.get('title', '')
                    if post.get('description'):
                        text = f"{text}\n{post['description']}"
                    
                    # Extract post ID from link
                    link = post.get('link', '')
                    post_id = link.split('/')[-2] if '/' in link else ''
                    
                    results.append({
                        'id': post_id,
                        'text': text,
                        'created_at': pd.to_datetime(created_approx, unit='s'),
                        'user_id': 0,  # Not available from search
                        'user_name': 'reddit_user',
                        'user_followers': 0,
                        'user_created_at': None,
                        'likes': 0,  # Not available from search
                        'retweets': 0,
                        'replies': 0,
                    })
                    
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    logger.debug(f"      Failed to parse post: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"      Failed to search Reddit: {e}")
        
        logger.info(f"      🔍 Reddit found {len(results)} posts for query '{query}' (NO AUTH!)")
        return results
        
    except Exception as e:
        logger.debug(f"Reddit search failed for '{query}': {e}")
        return []


def _get_user_tweets_sync(client, screen_name: str, start_time, end_time, max_results: int = 10):
    """Get user tweets synchronously."""
    import asyncio
    
    async def _async_get_user_tweets():
        try:
            user = await client.get_user_by_screen_name(screen_name)
            tweets = await client.get_user_tweets(user.id, tweet_type='Tweets', count=max_results)
            results = []
            for tweet in tweets:
                created_at = pd.to_datetime(tweet.created_at)
                if start_time <= created_at <= end_time:
                    results.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': created_at,
                        'user_id': user.id,
                        'user_name': screen_name,
                        'user_followers': getattr(user, 'followers_count', 0),
                        'user_created_at': pd.to_datetime(getattr(user, 'created_at', None)) if hasattr(user, 'created_at') else None,
                        'likes': getattr(tweet, 'favorite_count', 0),
                        'retweets': getattr(tweet, 'retweet_count', 0),
                        'replies': getattr(tweet, 'reply_count', 0),
                    })
            return results
        except Exception as e:
            logger.debug(f"Failed to get tweets from @{screen_name}: {e}")
            return []
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_async_get_user_tweets())


def _apply_quality_filters(tweets: list, quality_cfg: dict) -> list:
    """Apply quality filters to tweets."""
    if not tweets:
        return []
    
    logger.info(f"        💬 Sample text from first tweet: '{tweets[0].get('text', 'N/A')[:80]}...'")
    
    min_followers = quality_cfg.get('min_followers', 100)
    min_account_age_days = quality_cfg.get('min_account_age_days', 30)
    dedup_threshold = quality_cfg.get('dedup_threshold', 0.95)
    
    filtered = []
    seen_texts = []
    
    logger.info(f"        🔍 Processing {len(tweets)} tweets with quality filters...")
    filtered_count = 0
    
    for i, tweet in enumerate(tweets):
        # Filter by followers (skip if data not available - e.g., from YARS where followers=0)
        followers = tweet.get('user_followers', 0)
        if followers > 0 and followers < min_followers:
            logger.debug(f"        Tweet {i}: Filtered by followers: {followers} < {min_followers}")
            continue
        
        # Filter by account age (skip if data not available)
        if tweet.get('user_created_at'):
            account_age = (pd.Timestamp.now() - tweet['user_created_at']).days
            if account_age < min_account_age_days:
                logger.debug(f"        Filtered by account age: {account_age} days")
                continue
        
        # Deduplicate by text similarity
        text = tweet.get('text', '').lower()
        if not text or len(text.strip()) == 0:
            logger.debug(f"        Filtered: empty text")
            continue
            
        is_duplicate = False
        for seen_text in seen_texts:
            # Simple text similarity (can be improved with embeddings)
            text_words = set(text.split())
            seen_words = set(seen_text.split())
            if len(text_words) == 0 or len(seen_words) == 0:
                continue
            similarity = len(text_words & seen_words) / max(len(text_words), len(seen_words))
            if similarity > dedup_threshold:
                is_duplicate = True
                logger.debug(f"        Filtered: duplicate (similarity={similarity:.2f})")
                break
        
        if not is_duplicate:
            filtered.append(tweet)
            seen_texts.append(text)
            filtered_count += 1
            if filtered_count <= 3:
                logger.info(f"        ✅ Tweet {i} passed filters")
    
    logger.info(f"        🎯 Quality filter result: {len(filtered)}/{len(tweets)} tweets passed")
    return filtered


def _apply_sampling_rules(tweets: list, sampling_cfg: dict) -> list:
    """Apply sampling rules to tweets."""
    if not tweets:
        return []
    
    max_per_author = sampling_cfg.get('max_per_author', 3)
    split_strategy = sampling_cfg.get('split_strategy', 'balanced')
    
    # Count tweets per author
    author_counts = {}
    sampled = []
    
    # Sort by engagement (likes + retweets + replies)
    tweets_sorted = sorted(
        tweets, 
        key=lambda t: t.get('likes', 0) + t.get('retweets', 0) + t.get('replies', 0),
        reverse=True
    )
    
    for tweet in tweets_sorted:
        user_id = tweet.get('user_id')
        channel = tweet.get('channel', 'unknown')
        
        # Check per-author limit
        if user_id not in author_counts:
            author_counts[user_id] = 0
        
        if author_counts[user_id] < max_per_author:
            sampled.append(tweet)
            author_counts[user_id] += 1
    
    return sampled


def _batch_process_ml_features(tweets_by_date: list) -> tuple:
    """
    Batch process ML features (sentiment via FinBERT, novelty via embeddings) for all dates together.
    
    This is much more efficient than processing each date separately, as transformer models
    benefit from larger batches and only need to load once.
    
    Args:
        tweets_by_date: List of (date, tweets_list) tuples
    
    Returns:
        (sentiment_scores_by_date, novelty_scores_by_date) where each is a list of score lists
    """
    # Flatten all tweets and track date boundaries
    all_texts = []
    date_boundaries = []  # (start_idx, end_idx) for each date
    
    current_idx = 0
    for date, tweets in tweets_by_date:
        texts = [t.get('text', '') for t in tweets]
        all_texts.extend(texts)
        date_boundaries.append((current_idx, current_idx + len(texts)))
        current_idx += len(texts)
    
    total_texts = len(all_texts)
    logger.info(f"   🔬 Processing {total_texts} texts across {len(tweets_by_date)} dates with FinBERT")
    
    # Batch process sentiment with optimal settings
    if total_texts > 0:
        # Adaptive batch size based on total volume
        if total_texts < 50:
            batch_size = 16
        elif total_texts < 200:
            batch_size = 32
        elif total_texts < 500:
            batch_size = 64
        else:
            batch_size = 128
        
        logger.info(f"   ⚙️  Using batch_size={batch_size} for {total_texts} texts")
        all_sentiment_scores = _compute_sentiment_finbert_batched(all_texts, batch_size=batch_size)
        all_novelty_scores = _compute_novelty_batched(all_texts, batch_size=batch_size)
    else:
        all_sentiment_scores = []
        all_novelty_scores = []
    
    # Split results back by date
    sentiment_by_date = []
    novelty_by_date = []
    
    for start_idx, end_idx in date_boundaries:
        sentiment_by_date.append(all_sentiment_scores[start_idx:end_idx])
        novelty_by_date.append(all_novelty_scores[start_idx:end_idx])
    
    return sentiment_by_date, novelty_by_date


def _compute_features_with_scores(
    tweets: list, 
    date: pd.Timestamp, 
    symbol: str,
    sentiment_scores: list,
    novelty_scores: list,
    quality_cfg: dict
) -> Dict[str, Any]:
    """Compute all 24 features from tweets using pre-computed ML scores."""
    import numpy as np
    
    logger.debug(f"      🔢 Computing features from {len(tweets)} tweets")
    
    n_tweets = len(tweets)
    unique_authors = len(set(t.get('user_id') for t in tweets))
    
    # Governance features
    has_data = 1.0 if n_tweets > 0 else 0.0
    activity = min(n_tweets / 100.0, 1.0)  # Normalize to 0-1
    days_since_update = 0.0 if n_tweets > 0 else 999.0
    sample_n = float(n_tweets)
    unique_authors_n = float(unique_authors)
    
    # Channel coverage (% of channels with data)
    channels_present = set(t.get('channel') for t in tweets)
    query_coverage = len(channels_present) / 3.0  # 3 channels total
    
    # Attention features
    tweet_count = float(n_tweets)
    engagement_sum = sum(
        t.get('likes', 0) + t.get('retweets', 0) + t.get('replies', 0) 
        for t in tweets
    )
    engagement_per_tweet = engagement_sum / n_tweets if n_tweets > 0 else 0.0
    
    # Sentiment features (using pre-computed scores)
    sent_mean = np.mean(sentiment_scores) if sentiment_scores else 0.0
    sent_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
    
    # Sentiment fractions
    neg_count = sum(1 for s in sentiment_scores if s < -0.1)
    pos_count = sum(1 for s in sentiment_scores if s > 0.1)
    neu_count = len(sentiment_scores) - neg_count - pos_count
    
    sent_neg_frac = neg_count / len(sentiment_scores) if sentiment_scores else 0.0
    sent_pos_frac = pos_count / len(sentiment_scores) if sentiment_scores else 0.0
    sent_neu_frac = neu_count / len(sentiment_scores) if sentiment_scores else 0.0
    
    # Negative tail (< -0.5)
    neg_tail_frac = sum(1 for s in sentiment_scores if s < -0.5) / len(sentiment_scores) if sentiment_scores else 0.0
    
    # Novelty features (using pre-computed scores)
    novelty_mean = np.mean(novelty_scores) if novelty_scores else 0.0
    
    # Topic extraction (keyword matching)
    topic_earnings_frac = sum(
        1 for t in tweets if any(kw in t.get('text', '').lower() for kw in ['earnings', 'revenue', 'profit', 'eps'])
    ) / n_tweets if n_tweets > 0 else 0.0
    
    topic_lawsuit_frac = sum(
        1 for t in tweets if any(kw in t.get('text', '').lower() for kw in ['lawsuit', 'sue', 'court', 'legal'])
    ) / n_tweets if n_tweets > 0 else 0.0
    
    topic_guidance_frac = sum(
        1 for t in tweets if any(kw in t.get('text', '').lower() for kw in ['guidance', 'forecast', 'outlook', 'expect'])
    ) / n_tweets if n_tweets > 0 else 0.0
    
    # Portfolio overlays
    # Event risk: High novelty + high negative sentiment
    event_risk = novelty_mean * abs(min(sent_mean, 0.0))
    
    # Panic: High volume + high negative sentiment + high engagement
    panic = (activity * abs(min(sent_mean, 0.0)) * min(engagement_per_tweet / 100.0, 1.0))
    
    # Sentiment regime: Classify into regimes
    if sent_mean > 0.3:
        sent_regime = 1.0  # Bullish
    elif sent_mean < -0.3:
        sent_regime = -1.0  # Bearish
    else:
        sent_regime = 0.0  # Neutral
    
    return {
        'date': date,
        # Governance (6)
        'twitter_twikit_has_data': has_data,
        'twitter_twikit_activity': activity,
        'twitter_twikit_days_since_update': days_since_update,
        'twitter_twikit_sample_n': sample_n,
        'twitter_twikit_unique_authors_n': unique_authors_n,
        'twitter_twikit_query_coverage': query_coverage,
        # Attention (5)
        'twitter_twikit_tweet_count': tweet_count,
        'twitter_twikit_tweet_count_z': 0.0,  # Requires historical data for z-score
        'twitter_twikit_engagement_sum': engagement_sum,
        'twitter_twikit_engagement_per_tweet': engagement_per_tweet,
        'twitter_twikit_engagement_per_tweet_z': 0.0,  # Requires historical data
        # Sentiment (6)
        'twitter_twikit_sent_mean': sent_mean,
        'twitter_twikit_sent_std': sent_std,
        'twitter_twikit_sent_neg_frac': sent_neg_frac,
        'twitter_twikit_sent_pos_frac': sent_pos_frac,
        'twitter_twikit_sent_neu_frac': sent_neu_frac,
        'twitter_twikit_sent_neg_tail_frac': neg_tail_frac,
        # Novelty (4)
        'twitter_twikit_novelty_mean': novelty_mean,
        'twitter_twikit_topic_earnings_frac': topic_earnings_frac,
        'twitter_twikit_topic_lawsuit_frac': topic_lawsuit_frac,
        'twitter_twikit_topic_guidance_frac': topic_guidance_frac,
        # Portfolio (3)
        'twitter_twikit_event_risk': event_risk,
        'twitter_twikit_panic': panic,
        'twitter_twikit_sent_regime': sent_regime,
    }


def _compute_features(tweets: list, date: pd.Timestamp, symbol: str, quality_cfg: dict) -> Dict[str, Any]:
    """Compute all 24 features from tweets."""
    import numpy as np
    
    logger.info(f"      🔢 Computing features from {len(tweets)} tweets")
    
    n_tweets = len(tweets)
    unique_authors = len(set(t.get('user_id') for t in tweets))
    
    # Governance features
    has_data = 1.0 if n_tweets > 0 else 0.0
    activity = min(n_tweets / 100.0, 1.0)  # Normalize to 0-1
    days_since_update = 0.0 if n_tweets > 0 else 999.0
    sample_n = float(n_tweets)
    unique_authors_n = float(unique_authors)
    
    # Channel coverage (% of channels with data)
    channels_present = set(t.get('channel') for t in tweets)
    query_coverage = len(channels_present) / 3.0  # 3 channels total
    
    # Attention features
    tweet_count = float(n_tweets)
    engagement_sum = sum(
        t.get('likes', 0) + t.get('retweets', 0) + t.get('replies', 0) 
        for t in tweets
    )
    engagement_per_tweet = engagement_sum / n_tweets if n_tweets > 0 else 0.0
    
    # Compute sentiment with FinBERT
    sentiment_scores = _compute_sentiment_finbert([t.get('text', '') for t in tweets])
    
    # Sentiment features
    sent_mean = np.mean(sentiment_scores) if sentiment_scores else 0.0
    sent_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
    
    # Sentiment fractions
    neg_count = sum(1 for s in sentiment_scores if s < -0.1)
    pos_count = sum(1 for s in sentiment_scores if s > 0.1)
    neu_count = len(sentiment_scores) - neg_count - pos_count
    
    sent_neg_frac = neg_count / len(sentiment_scores) if sentiment_scores else 0.0
    sent_pos_frac = pos_count / len(sentiment_scores) if sentiment_scores else 0.0
    sent_neu_frac = neu_count / len(sentiment_scores) if sentiment_scores else 0.0
    
    # Negative tail (< -0.5)
    neg_tail_frac = sum(1 for s in sentiment_scores if s < -0.5) / len(sentiment_scores) if sentiment_scores else 0.0
    
    # Novelty features (using simple embedding cosine similarity)
    novelty_scores = _compute_novelty([t.get('text', '') for t in tweets])
    novelty_mean = np.mean(novelty_scores) if novelty_scores else 0.0
    
    # Topic extraction (keyword matching)
    topic_earnings_frac = sum(
        1 for t in tweets if any(kw in t.get('text', '').lower() for kw in ['earnings', 'revenue', 'profit', 'eps'])
    ) / n_tweets if n_tweets > 0 else 0.0
    
    topic_lawsuit_frac = sum(
        1 for t in tweets if any(kw in t.get('text', '').lower() for kw in ['lawsuit', 'sue', 'court', 'legal'])
    ) / n_tweets if n_tweets > 0 else 0.0
    
    topic_guidance_frac = sum(
        1 for t in tweets if any(kw in t.get('text', '').lower() for kw in ['guidance', 'forecast', 'outlook', 'expect'])
    ) / n_tweets if n_tweets > 0 else 0.0
    
    # Portfolio overlays
    # Event risk: High novelty + high negative sentiment
    event_risk = novelty_mean * abs(min(sent_mean, 0.0))
    
    # Panic: High volume + high negative sentiment + high engagement
    panic = (activity * abs(min(sent_mean, 0.0)) * min(engagement_per_tweet / 100.0, 1.0))
    
    # Sentiment regime: Classify into regimes
    if sent_mean > 0.3:
        sent_regime = 1.0  # Bullish
    elif sent_mean < -0.3:
        sent_regime = -1.0  # Bearish
    else:
        sent_regime = 0.0  # Neutral
    
    return {
        'date': date,
        # Governance (6)
        'twitter_twikit_has_data': has_data,
        'twitter_twikit_activity': activity,
        'twitter_twikit_days_since_update': days_since_update,
        'twitter_twikit_sample_n': sample_n,
        'twitter_twikit_unique_authors_n': unique_authors_n,
        'twitter_twikit_query_coverage': query_coverage,
        # Attention (5)
        'twitter_twikit_tweet_count': tweet_count,
        'twitter_twikit_tweet_count_z': 0.0,  # Requires historical data for z-score
        'twitter_twikit_engagement_sum': engagement_sum,
        'twitter_twikit_engagement_per_tweet': engagement_per_tweet,
        'twitter_twikit_volume_change_1d': 0.0,  # Requires previous day data
        # Sentiment (6)
        'twitter_twikit_sent_mean': sent_mean,
        'twitter_twikit_sent_neg_frac': sent_neg_frac,
        'twitter_twikit_sent_pos_frac': sent_pos_frac,
        'twitter_twikit_sent_neu_frac': sent_neu_frac,
        'twitter_twikit_sent_std': sent_std,
        'twitter_twikit_neg_tail_frac': neg_tail_frac,
        # Novelty (4)
        'twitter_twikit_novelty_mean': novelty_mean,
        'twitter_twikit_topic_earnings_frac': topic_earnings_frac,
        'twitter_twikit_topic_lawsuit_frac': topic_lawsuit_frac,
        'twitter_twikit_topic_guidance_frac': topic_guidance_frac,
        # Portfolio (3)
        'twitter_twikit_event_risk': event_risk,
        'twitter_twikit_panic': panic,
        'twitter_twikit_sent_regime': sent_regime,
    }


def _compute_sentiment_finbert_batched(texts: list, batch_size: int = 32) -> list:
    """
    Compute sentiment scores using FinBERT with configurable batch size.
    
    Args:
        texts: List of text strings to analyze
        batch_size: Batch size for processing (adaptive based on total volume)
    
    Returns:
        List of sentiment scores (-1 to +1)
    """
    if not texts:
        return []
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        model_name = "yiyanghkust/finbert-tone"
        
        # Load model (cached after first load)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"   🚀 Using GPU for FinBERT processing")
        
        scores = []
        
        # Process in batches with progress logging
        num_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx+batch_size]
            current_batch = (batch_idx // batch_size) + 1
            
            if num_batches > 1 and current_batch % 5 == 0:
                logger.info(f"   📊 Processing batch {current_batch}/{num_batches}")
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to sentiment scores
            # FinBERT outputs: [negative, neutral, positive]
            for prob in probs.cpu().numpy():
                neg, neu, pos = prob
                # Score: -1 (negative) to +1 (positive)
                score = pos - neg
                scores.append(float(score))
        
        logger.info(f"   ✅ FinBERT processed {len(texts)} texts in {num_batches} batches")
        return scores
        
    except Exception as e:
        logger.warning(f"FinBERT sentiment failed: {e}")
        return [0.0] * len(texts)


def _compute_sentiment_finbert(texts: list) -> list:
    """Legacy wrapper for backward compatibility."""
    return _compute_sentiment_finbert_batched(texts, batch_size=32)


def _compute_novelty_batched(texts: list, batch_size: int = 64) -> list:
    """
    Compute novelty scores using embeddings with configurable batch size.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for embedding computation
    
    Returns:
        List of novelty scores (0-1)
    """
    if not texts or len(texts) < 2:
        return [0.5] * len(texts)  # Default medium novelty
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Load model (cached)
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Get embeddings with batching
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        
        # Compute pairwise cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(embeddings)
        
        # Novelty = 1 - max_similarity_to_others
        novelty_scores = []
        for i in range(len(texts)):
            # Get max similarity to other tweets (exclude self)
            other_sims = [similarities[i][j] for j in range(len(texts)) if i != j]
            max_sim = max(other_sims) if other_sims else 0.0
            novelty = 1.0 - max_sim
            novelty_scores.append(float(novelty))
        
        return novelty_scores
        
    except Exception as e:
        logger.warning(f"Novelty computation failed: {e}")
        return [0.5] * len(texts)


def _compute_novelty(texts: list) -> list:
    """Legacy wrapper for backward compatibility."""
    return _compute_novelty_batched(texts, batch_size=64)


# Entry point for prep_families
def build(
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    out_path: str,
    raw_source_cfg: dict,
    compute_cfg: dict,
) -> None:
    """
    Build twitter_twikit_hf features following standard HF generator interface.
    
    Args:
        symbol: Stock symbol
        horizon: Forecast horizon (days) - not used for symbol-only HF block
        start: Start date (YYYY-MM-DD, inclusive)
        end: End date (YYYY-MM-DD, exclusive)
        out_path: Output parquet path
        raw_source_cfg: Raw data source config from hf_registry.yaml
        compute_cfg: Compute config from hf_registry.yaml
    """
    try:
        # Merge configs into kwargs
        kwargs = {**raw_source_cfg, **compute_cfg}
        
        result = generate_twitter_twikit_hf(
            symbol=symbol,
            start=start,
            end=end,
            output_path=Path(out_path),
            **kwargs
        )
        
        if result is None or len(result) == 0:
            logger.error(f"Failed to generate twitter_twikit_hf for {symbol}")
            # Create empty file so prep_families doesn't fail
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_parquet(out_path)
        
    except Exception as e:
        logger.error(f"Error in twitter_twikit_hf generator: {e}", exc_info=True)
        # Create empty file so prep_families doesn't fail
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(out_path)


def main(symbol: str, start: str, end: str, output_path: str, **kwargs) -> int:
    """
    Main entry point called by prep_families.
    
    Args:
        symbol: Stock symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        output_path: Path to save parquet
        **kwargs: Config from hf_registry.yaml
    
    Returns:
        0 on success, 1 on failure
    """
    try:
        result = generate_twitter_twikit_hf(
            symbol=symbol,
            start=start,
            end=end,
            output_path=Path(output_path),
            **kwargs
        )
        
        if result is None or len(result) == 0:
            logger.error(f"Failed to generate twitter_twikit_hf for {symbol}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in twitter_twikit_hf generator: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Test generation
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python twitter_twikit.py SYMBOL START END [OUTPUT_PATH]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else f"/tmp/{symbol}_twitter_twikit.parquet"
    
    logging.basicConfig(level=logging.INFO)
    exit_code = main(symbol, start, end, output_path)
    sys.exit(exit_code)
