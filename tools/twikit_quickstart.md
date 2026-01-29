# Twikit Quick Start - Three-Channel Twitter/X Data Collection

## Installation Status

✅ **Twikit v2.3.3** installed in `.venv/lib/python3.12/site-packages`
- GitHub: https://github.com/d60/twikit
- All dependencies installed: httpx, beautifulsoup4, lxml, pyotp, etc.

## Import in Python

```python
import sys
sys.path.insert(0, '.venv/lib/python3.12/site-packages')
from twikit import Client
```

Or use your project's existing Python environment that has access to `.venv`.

## Basic Authentication

```python
from twikit import Client

# Initialize client
client = Client(language='en-US')

# Option 1: Login with username/password
await client.login(
    auth_info_1='username_or_email',
    auth_info_2='email_or_phone',
    password='your_password'
)

# Option 2: Load cookies from file (recommended for production)
client.load_cookies('path/to/cookies.json')

# Save cookies for reuse
client.save_cookies('path/to/cookies.json')
```

## Three-Channel Implementation

### Channel A: Cashtag Intent ($TICKER)

```python
async def collect_channel_a(client, symbol):
    """
    Highest precision - direct market intent from traders.
    Query: $AAPL
    Expected: 50-200 tweets/day for liquid stocks
    """
    query = f'${symbol}'
    tweets = await client.search_tweet(query, product='Latest', count=100)
    
    # Process results
    results = []
    for tweet in tweets:
        results.append({
            'id': tweet.id,
            'text': tweet.text,
            'created_at': tweet.created_at,
            'user_id': tweet.user.id,
            'user_name': tweet.user.name,
            'followers': tweet.user.followers_count,
            'likes': tweet.favorite_count,
            'retweets': tweet.retweet_count,
            'replies': tweet.reply_count,
            'is_retweet': tweet.is_retweet,
            'is_quote': tweet.is_quoted,
        })
    
    return results
```

### Channel B: Name Intent (Company + Finance Context)

```python
async def collect_channel_b(client, symbol, company_name):
    """
    Higher recall - broader discussion with finance filters.
    Query: Apple (stock OR earnings OR guidance OR $AAPL)
    Expected: 100-500 tweets/day
    Note: Requires strong quality filters due to noise
    """
    query = f'{company_name} (stock OR earnings OR guidance OR ${symbol})'
    tweets = await client.search_tweet(query, product='Latest', count=100)
    
    # Apply quality filters (see below)
    filtered = apply_quality_filters(tweets)
    
    return filtered
```

### Channel C: Curated Accounts (Official Sources)

```python
async def collect_channel_c(client, curated_accounts):
    """
    Highest quality - authoritative sources only.
    Sources: @Apple_IR, @Reuters, @Bloomberg, sector analysts
    Expected: 1-20 tweets/day
    """
    all_tweets = []
    
    for screen_name in curated_accounts:
        # Get user
        user = await client.get_user_by_screen_name(screen_name)
        
        # Get recent tweets
        tweets = await client.get_user_tweets(
            user.id,
            tweet_type='Tweets',  # Excludes replies/retweets
            count=20
        )
        
        all_tweets.extend(tweets)
    
    return all_tweets
```

## Sampling Rules (Anti-Bias)

### Rule 1: Cap Per Author

```python
from collections import defaultdict

def cap_per_author(tweets, max_per_author=3):
    """Prevent single influencer from dominating signal."""
    author_counts = defaultdict(int)
    filtered = []
    
    for tweet in tweets:
        if author_counts[tweet.user.id] < max_per_author:
            filtered.append(tweet)
            author_counts[tweet.user.id] += 1
    
    return filtered
```

### Rule 2: Split Sample (50% Top + 50% Random)

```python
import random

def split_sample(tweets):
    """Balance 'what's famous' with 'what's changing'."""
    # Sort by engagement
    engagement = lambda t: t.favorite_count + t.retweet_count + t.reply_count
    sorted_tweets = sorted(tweets, key=engagement, reverse=True)
    
    # Take 50% top + 50% random
    n_half = len(tweets) // 2
    top_half = sorted_tweets[:n_half]
    random_half = random.sample(tweets, min(n_half, len(tweets)))
    
    # Combine and deduplicate
    final = list({t.id: t for t in (top_half + random_half)}.values())
    
    return final
```

### Rule 3: Separate Originals vs Retweets

```python
def filter_tweet_types(tweets, include_retweets=False, include_replies=False):
    """
    Originals + Quote-tweets: Most informative, full text ingestion
    Pure retweets: Count as amplification signal only
    Replies: Often noisy, use selectively
    """
    filtered = []
    
    for tweet in tweets:
        # Skip pure retweets
        if tweet.is_retweet and not include_retweets:
            continue
        
        # Skip replies
        if tweet.in_reply_to_status_id and not include_replies:
            continue
        
        # Keep originals and quote-tweets
        filtered.append(tweet)
    
    return filtered
```

## Quality Filters

### User-Level Credibility

```python
from datetime import datetime

def apply_user_filters(tweets, min_followers=100, min_account_age_days=30, max_posts_per_day=100):
    """Remove spam/bot accounts."""
    filtered = []
    
    for tweet in tweets:
        user = tweet.user
        
        # Check minimum followers
        if user.followers_count < min_followers:
            continue
        
        # Check account age
        account_age = (datetime.now() - user.created_at_datetime).days
        if account_age < min_account_age_days:
            continue
        
        # Check posting rate (flag bots)
        posts_per_day = user.statuses_count / max(1, account_age)
        if posts_per_day > max_posts_per_day:
            continue
        
        filtered.append(tweet)
    
    return filtered
```

### Text-Level Spam Filters

```python
def apply_text_filters(tweets, spam_keywords=['giveaway', 'airdrop', 'promo', 'contest', 'win', 'free'],
                       max_hashtags=5, max_links=3):
    """Remove spam content."""
    filtered = []
    
    for tweet in tweets:
        text_lower = tweet.text.lower()
        
        # Check spam keywords
        if any(kw in text_lower for kw in spam_keywords):
            continue
        
        # Count hashtags
        hashtag_count = tweet.text.count('#')
        if hashtag_count > max_hashtags:
            continue
        
        # Count links
        link_count = tweet.text.count('http')
        if link_count > max_links:
            continue
        
        filtered.append(tweet)
    
    return filtered
```

### Deduplication

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_tweets(tweets, threshold=0.95):
    """Remove near-duplicate tweets."""
    if len(tweets) <= 1:
        return tweets
    
    texts = [t.text for t in tweets]
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(texts)
    
    # Keep first occurrence of unique tweets
    keep_indices = [0]  # Always keep first
    
    for i in range(1, len(tweets)):
        # Compare with already kept tweets
        kept_tfidf = tfidf[keep_indices]
        current_tfidf = tfidf[i:i+1]
        
        sims = cosine_similarity(current_tfidf, kept_tfidf)
        
        # Keep if not too similar to any existing
        if sims.max() < threshold:
            keep_indices.append(i)
    
    return [tweets[i] for i in keep_indices]
```

## Complete Three-Channel Pipeline

```python
async def collect_three_channel_data(symbol, company_name, curated_accounts, date):
    """
    Complete three-channel data collection with all filters applied.
    Returns: List of filtered, deduplicated tweets ready for feature extraction.
    """
    from twikit import Client
    
    # Initialize and authenticate
    client = Client(language='en-US')
    client.load_cookies('path/to/cookies.json')
    
    # Collect from all three channels
    channel_a = await collect_channel_a(client, symbol)
    channel_b = await collect_channel_b(client, symbol, company_name)
    channel_c = await collect_channel_c(client, curated_accounts)
    
    # Merge all channels
    all_tweets = channel_a + channel_b + channel_c
    
    # Apply sampling rules
    all_tweets = filter_tweet_types(all_tweets, include_retweets=False, include_replies=False)
    all_tweets = cap_per_author(all_tweets, max_per_author=3)
    all_tweets = split_sample(all_tweets)
    
    # Apply quality filters
    all_tweets = apply_user_filters(all_tweets, min_followers=100, min_account_age_days=30)
    all_tweets = apply_text_filters(all_tweets)
    all_tweets = deduplicate_tweets(all_tweets, threshold=0.95)
    
    # Compute features (see feature computation section)
    features = compute_features(all_tweets, symbol, date)
    
    return features
```

## Available Tweet Fields

```python
# Tweet object fields
tweet.id                    # Tweet ID
tweet.text                  # Full text content
tweet.created_at            # Timestamp
tweet.favorite_count        # Likes
tweet.retweet_count         # Retweets
tweet.reply_count           # Replies
tweet.view_count            # Views (if available)
tweet.is_retweet            # Pure retweet flag
tweet.is_quoted             # Quote-tweet flag
tweet.in_reply_to_status_id # Reply flag
tweet.lang                  # Language code

# User object fields
tweet.user.id               # User ID
tweet.user.name             # Display name
tweet.user.screen_name      # @handle
tweet.user.followers_count  # Follower count
tweet.user.statuses_count   # Total tweets
tweet.user.created_at_datetime  # Account creation date
tweet.user.verified         # Legacy verified
tweet.user.is_blue_verified # Blue checkmark
```

## Rate Limits & Best Practices

1. **Rate Limiting:**
   - Twitter/X has strict rate limits
   - Implement exponential backoff on errors
   - Cache results to minimize API calls

2. **Authentication:**
   - Save cookies after login for reuse
   - Rotate accounts if collecting at scale
   - Handle 2FA/CAPTCHA challenges

3. **Error Handling:**
   - Catch network errors, timeouts
   - Handle suspended/deleted accounts gracefully
   - Log failures for debugging

4. **Caching:**
   - Cache raw tweets to `data/cache/twitter_twikit/`
   - Store by date and symbol for easy retrieval
   - Reprocess features without re-fetching

## Next Steps for twitter_twikit Family

1. ✅ Install twikit and dependencies
2. 🔄 Set up authentication (get Twitter cookies/credentials)
3. 🔄 Implement three-channel collection in `tools/hf_generators/twitter_twikit.py`
4. 🔄 Add FinBERT sentiment scoring
5. 🔄 Add sentence-transformers for novelty detection
6. 🔄 Compute all 24 features per date
7. 🔄 Update `enabled: true` in `tools/hf_registry.yaml`
8. 🔄 Test with AAPL, NVDA, TSLA

---

**Last Updated:** January 26, 2026  
**Twikit Version:** 2.3.3  
**Status:** Installed and ready for implementation
