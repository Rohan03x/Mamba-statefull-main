import time
import threading
import random


class TokenBucket:
    def __init__(self, rate_per_min: float = 5.0, burst: int = 5):
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.refill = float(rate_per_min) / 60.0
        self.last = time.time()
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            now = time.time()
            dt = now - self.last
            self.last = now
            # refill
            self.tokens = min(self.capacity, self.tokens + dt * self.refill)
            need = 1.0
            if self.tokens >= need:
                self.tokens -= need
                return
            # sleep until we have enough tokens
            missing = need - self.tokens
            sleep_s = missing / max(self.refill, 1e-9)
        time.sleep(max(0.0, sleep_s))
        with self.lock:
            self.tokens = max(0.0, self.tokens - 0.0)


_bucket = TokenBucket(rate_per_min=5.0, burst=5)
_retries = 0
_backoff_max = 0.0


def backoff_try(fn, *args, **kwargs):
    delay = 1.0
    for _ in range(5):
        _bucket.take()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "temporar" in msg:
                global _retries, _backoff_max
                _retries += 1
                jitter = random.random()
                _backoff_max = max(_backoff_max, delay + jitter)
                time.sleep(delay + jitter)
                delay *= 2.0
                continue
            raise
    raise RuntimeError("rate/backoff exhausted")


def get_stats() -> dict:
    return {
        "retries": _retries,
        "backoff_max": _backoff_max,
    }


def reset_stats() -> None:
    global _retries, _backoff_max
    _retries = 0
    _backoff_max = 0.0
