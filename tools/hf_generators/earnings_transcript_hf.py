"""Earnings Transcript HF Generator.

Fetches earnings call transcripts via defeatbeta-api and scores them using a
transformer (FinBERT-style) model.

Leak-safe design:
- Only reads transcripts published within [start, end)
- Aligns signals to NYSE sessions in [start, end)
- No future peeking: transcript date is treated as its publication/report date

Output schema (unprefixed; prep_families will prefix when persisting):
- date: session label
- score: sentiment score (event-weighted)
- conf: confidence (event-weighted)
- has_data: 1.0 on earnings session; 0.0 otherwise

Important:
- We intentionally keep this event-driven. By default we extend each event over
    a small number of sessions with a fast decay (3-4 sessions) so the model sees
    short-lived impact beyond the earnings day.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)


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
    Build earnings transcript signal for a symbol/horizon/date range.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        horizon: Forecast horizon in trading days
        start: Start date (ISO format, inclusive)
        end: End date (ISO format, exclusive)
        out_path: Output parquet path
        raw_source_cfg: Dict with keys:
            - transcript_dir: Path to raw transcript data
            - transcript_format: 'parquet' | 'csv' | 'json' | 'txt'
            - date_col: Column name for call date
            - text_col: Column name for transcript text
        compute_cfg: Dict with keys:
            - model: HF model name (default: 'ProsusAI/finbert')
            - batch_size: Batch size for inference (default: 8)
            - max_length: Max token length (default: 1024)
            - decay_days: Days to decay signal after call (default: 90)
    
    Returns:
        None (writes parquet to out_path)
    """
    LOGGER.info(
        f"Building earnings_transcript_hf: {symbol} h{horizon} [{start}, {end})"
    )
    
    # Parse dates
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Load raw transcript data (ONLY within date range)
    transcript_df = _load_raw_transcripts(symbol, start_dt, end_dt, raw_source_cfg)
    
    if transcript_df.empty:
        LOGGER.warning(
            f"No earnings transcripts found for {symbol} in [{start}, {end})"
        )
        # Create empty signal
        signal_df = _create_empty_signal(start_dt, end_dt)
    else:
        # Compute features using HF model
        transcript_df = _compute_transcript_features(transcript_df, compute_cfg)
        
        # Align to bar calendar with decay
        signal_df = _align_to_bars(
            transcript_df, start_dt, end_dt, compute_cfg
        )
    
    # Validate output format
    _validate_signal(signal_df)
    
    # Write to parquet
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_parquet(out_path, index=False)
    
    LOGGER.info(
        f"✅ Wrote earnings_transcript_hf signal: {len(signal_df)} bars → {out_path}"
    )


def _load_raw_transcripts(
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    cfg: dict,
) -> pd.DataFrame:
    """Load actual earnings call transcripts from defeatbeta-api."""
    try:
        from defeatbeta_api.data.ticker import Ticker
        
        LOGGER.info(f"📞 Fetching earnings call transcripts for {symbol} from defeatbeta-api...")
        
        ticker = Ticker(symbol)
        transcripts_obj = ticker.earning_call_transcripts()
        transcripts_df = transcripts_obj.get_transcripts_list()
        
        if isinstance(transcripts_df, pd.DataFrame) and not transcripts_df.empty:
            # Convert report_date to datetime
            transcripts_df['date'] = pd.to_datetime(transcripts_df['report_date'])
            
            LOGGER.info(f"📊 Total transcripts from API: {len(transcripts_df)}")
            LOGGER.info(f"📅 Date range: {start_dt} to {end_dt}")
            LOGGER.info(f"📅 Transcript dates: {transcripts_df['date'].min()} to {transcripts_df['date'].max()}")
            
            # Filter to date range
            transcripts_df = transcripts_df[
                (transcripts_df['date'] >= start_dt) & 
                (transcripts_df['date'] < end_dt)
            ]
            
            LOGGER.info(f"📊 Transcripts after date filter: {len(transcripts_df)}")
            
            if not transcripts_df.empty:
                # Extract text from transcripts column (numpy array of paragraph dicts)
                texts = []
                for idx, row in transcripts_df.iterrows():
                    transcript_paragraphs = row['transcripts']
                    if isinstance(transcript_paragraphs, (list, np.ndarray)):
                        # Combine all paragraphs into one text
                        full_text = ' '.join(
                            para.get('content', '') 
                            for para in transcript_paragraphs 
                            if isinstance(para, dict) and para.get('content', '')
                        )
                        texts.append(full_text[:10000])  # Limit to first 10k chars
                    else:
                        texts.append('')
                
                LOGGER.info(f"📝 Extracted {len(texts)} text samples, avg length: {sum(len(t) for t in texts) / max(len(texts), 1):.0f} chars")
                
                result_df = pd.DataFrame({
                    'date': transcripts_df['date'].values,
                    'text': texts
                })
                
                # Remove empty transcripts
                before_filter = len(result_df)
                result_df = result_df[result_df['text'].str.len() > 100]
                after_filter = len(result_df)
                
                LOGGER.info(f"📊 After length filter (>100 chars): {after_filter}/{before_filter} transcripts")
                
                LOGGER.info(f"✅ Loaded {len(result_df)} earnings call transcripts")
                return result_df
            else:
                LOGGER.warning(f"No transcripts found in date range [{start_dt}, {end_dt})")
                return pd.DataFrame()
        else:
            LOGGER.warning(f"No transcripts available from defeatbeta-api")
            return pd.DataFrame()
        
    except ImportError:
        LOGGER.warning("defeatbeta-api not available, cannot fetch transcripts")
        return pd.DataFrame()
    except Exception as e:
        LOGGER.warning(f"Failed to fetch transcripts: {e}")
        return pd.DataFrame()


def _compute_transcript_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute sentiment features from transcripts using FinBERT."""
    model_name = cfg.get("model", "yiyanghkust/finbert-tone")
    batch_size = cfg.get("batch_size", 4)
    max_length = cfg.get("max_length", 512)
    
    LOGGER.info(f"Computing transcript sentiment with model: {model_name}")
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    # Load FinBERT model (use safetensors to avoid torch.load vulnerability)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=True
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        LOGGER.info("Using GPU for FinBERT inference")
    
    scores = []
    confs = []
    
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_texts = df["text"].iloc[i:i+batch_size].tolist()
        
        # Tokenize (truncate long transcripts)
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
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
            # Confidence: max probability
            conf = max(prob)
            scores.append(score)
            confs.append(conf)
    
    df["score"] = scores
    df["conf"] = confs
    
    LOGGER.info(f"✅ Computed sentiment for {len(df)} transcripts")
    
    return df


def _align_to_bars(
    df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    cfg: dict,
) -> pd.DataFrame:
    """Align transcript features to NYSE sessions with event-aware two-phase decay.

    Design goals (leak-safe):
    - Align events to trading sessions.
    - Start influence no earlier than the next session after transcript report date.
    - Apply a two-component decay: fast assimilation + slow narrative persistence.
    - Hard reset at the next earnings transcript event (no cross-quarter stacking).
    - Optionally adapt half-lives by event strength and sign.
    """

    import exchange_calendars as xcals  # type: ignore

    start_dt = pd.Timestamp(start_dt).normalize()
    end_dt = pd.Timestamp(end_dt).normalize()
    if end_dt <= start_dt:
        return pd.DataFrame(columns=["date", "score", "conf", "has_data"])

    cal = xcals.get_calendar("XNYS")
    cal_first = pd.Timestamp(cal.first_session).normalize()
    cal_last = pd.Timestamp(cal.last_session).normalize()

    # Guard against requests that extend beyond the calendar's supported range.
    # (exchange_calendars raises if you pass a date earlier than first_session or
    # later than last_session.)
    if end_dt <= cal_first:
        return pd.DataFrame(columns=["date", "score", "conf", "has_data"])

    # Clip start/end for session computations only; we still emit a daily calendar
    # for the original [start_dt, end_dt) below.
    start_for_sessions = max(start_dt, cal_first)
    start_sess = pd.Timestamp(cal.date_to_session(start_for_sessions, direction="next"))

    if end_dt > cal_last:
        # Treat end as beyond the calendar; include sessions through last_session.
        last_sess = pd.Timestamp(cal.last_session)
    else:
        end_sess_excl = pd.Timestamp(cal.date_to_session(end_dt, direction="next"))
        # sessions are inclusive, so we use the previous session of end-exclusive
        try:
            last_sess = pd.Timestamp(cal.previous_session(end_sess_excl))
        except Exception:
            # If end_sess_excl is the first known session, there are no sessions in-range.
            return pd.DataFrame(columns=["date", "score", "conf", "has_data"])

    if last_sess < start_sess:
        return pd.DataFrame(columns=["date", "score", "conf", "has_data"])

    sessions = pd.DatetimeIndex(cal.sessions_in_range(start_sess, last_sess)).tz_localize(None)

    # ---------------------------------------------------------------------
    # Config defaults
    # ---------------------------------------------------------------------
    def _half_life_to_tau(half_life: float) -> float:
        half_life = float(max(1e-6, half_life))
        return half_life / float(np.log(2.0))

    fast_half_life = float(cfg.get("fast_half_life_sessions", 4.0))
    fast_duration = int(cfg.get("fast_duration_sessions", 10))
    slow_half_life = float(cfg.get("slow_half_life_sessions", 35.0))
    fast_weight = float(cfg.get("fast_weight", 0.65))
    slow_weight = float(cfg.get("slow_weight", 0.35))

    # Confidence-weighted decay scaling.
    scale_min = float(cfg.get("decay_scale_min", 0.5))
    scale_max = float(cfg.get("decay_scale_max", 1.5))
    z_norm_div = float(cfg.get("strength_z_divisor", 2.0))

    # Directional asymmetry (slow tail only).
    neg_slow_mult = float(cfg.get("neg_slow_half_life_mult", 1.3))
    pos_slow_mult = float(cfg.get("pos_slow_half_life_mult", 0.85))

    # Shift alignment to next trading session (conservative, leak-safe).
    shift_to_next = bool(cfg.get("shift_to_next_session", True))

    # Normalize transcript dates.
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(sessions)}
    score = np.zeros(len(sessions), dtype=float)
    conf = np.zeros(len(sessions), dtype=float)
    has_data = np.zeros(len(sessions), dtype=float)

    # Event-only (undecayed) features for auditability / future retuning.
    event_score = np.zeros(len(sessions), dtype=float)
    event_conf = np.zeros(len(sessions), dtype=float)
    event_sentiment_z = np.zeros(len(sessions), dtype=float)
    event_strength = np.zeros(len(sessions), dtype=float)

    # Map transcripts to sessions, then aggregate per session.
    events: list[tuple[pd.Timestamp, float, float]] = []
    for row in df.sort_values("date").itertuples(index=False):
        d = getattr(row, "date", None)
        if not isinstance(d, pd.Timestamp):
            continue
        d = pd.Timestamp(d).normalize()
        if d < cal_first or d > cal_last:
            continue
        try:
            sess = pd.Timestamp(cal.date_to_session(d, direction="next")).normalize()
            if shift_to_next:
                sess = pd.Timestamp(cal.next_session(sess)).normalize()
        except Exception:
            continue
        s_val = float(getattr(row, "score", 0.0) or 0.0)
        c_val = float(getattr(row, "conf", 0.0) or 0.0)
        if not np.isfinite(s_val) or not np.isfinite(c_val):
            continue
        events.append((sess, s_val, float(np.clip(c_val, 0.0, 1.0))))

    if not events:
        session_df = pd.DataFrame({
            "date": sessions,
            "score": score,
            "conf": conf,
            "has_data": has_data,
            "earnings_transcript_hf_event_score": event_score,
            "earnings_transcript_hf_event_conf": event_conf,
            "earnings_transcript_hf_event_sentiment_z": event_sentiment_z,
            "earnings_transcript_hf_event_strength": event_strength,
        })
    else:
        events_df = pd.DataFrame(events, columns=["session", "score", "conf"])
        # Combine multiple transcripts that map to the same session (weighted by conf).
        def _wavg(group: pd.DataFrame) -> pd.Series:
            w = group["conf"].to_numpy(dtype=float)
            s = group["score"].to_numpy(dtype=float)
            w_sum = float(np.sum(w))
            score_avg = float(np.sum(s * w) / w_sum) if w_sum > 0 else float(np.mean(s))
            conf_agg = float(np.clip(np.max(w), 0.0, 1.0))
            return pd.Series({"score": score_avg, "conf": conf_agg})

        events_df = events_df.groupby("session", as_index=False).apply(_wavg, include_groups=False).reset_index(drop=True)
        events_df["session"] = pd.to_datetime(events_df["session"]).dt.normalize()
        events_df = events_df.sort_values("session").reset_index(drop=True)

        # Precompute the end boundary for each event (hard reset at next event).
        next_sessions = list(events_df["session"].iloc[1:]) + [None]

        # Online (past-only) expanding z-score on event scores.
        prev_scores: list[float] = []
        for (sess, s_val, c_val), next_sess in zip(events_df[["session", "score", "conf"]].itertuples(index=False, name=None), next_sessions):
            base = pos.get(pd.Timestamp(sess).normalize())
            if base is None:
                continue

            # Compute sentiment z-score using only prior events.
            z = 0.0
            if len(prev_scores) >= 5:
                mu = float(np.mean(prev_scores))
                sd = float(np.std(prev_scores, ddof=0))
                if sd > 1e-12:
                    z = float((float(s_val) - mu) / sd)
            prev_scores.append(float(s_val))

            strength = float(abs(z) * float(c_val))
            strength_norm = float(min(1.0, strength / max(1e-6, z_norm_div)))
            scale = float(np.clip(scale_min + strength_norm, scale_min, scale_max))

            slow_hl_adj = slow_half_life
            if float(s_val) < 0:
                slow_hl_adj *= neg_slow_mult
            elif float(s_val) > 0:
                slow_hl_adj *= pos_slow_mult
            slow_hl_adj *= scale
            fast_hl_adj = fast_half_life * scale

            tau_fast = _half_life_to_tau(fast_hl_adj)
            tau_slow = _half_life_to_tau(slow_hl_adj)

            # Mark event-only series.
            has_data[base] = 1.0
            event_score[base] = float(s_val)
            event_conf[base] = float(c_val)
            event_sentiment_z[base] = float(z)
            event_strength[base] = float(strength)

            # Hard reset at next event session.
            end_idx = len(sessions)
            if next_sess is not None:
                nxt = pos.get(pd.Timestamp(next_sess).normalize())
                if nxt is not None:
                    end_idx = min(end_idx, nxt)

            # Apply two-component decay from base until end_idx.
            for t in range(0, max(0, end_idx - base)):
                w_fast = 0.0
                if t < max(1, fast_duration):
                    w_fast = fast_weight * float(np.exp(-float(t) / tau_fast))
                w_slow = slow_weight * float(np.exp(-float(t) / tau_slow))
                w_total = w_fast + w_slow
                if w_total <= 0.0:
                    continue
                idx = base + t
                score[idx] += float(s_val) * w_total
                conf[idx] += float(c_val) * w_total

        conf = np.clip(conf, 0.0, 1.0)

        session_df = pd.DataFrame({
            "date": sessions,
            "score": score,
            "conf": conf,
            "has_data": has_data,
            "earnings_transcript_hf_event_score": event_score,
            "earnings_transcript_hf_event_conf": event_conf,
            "earnings_transcript_hf_event_sentiment_z": event_sentiment_z,
            "earnings_transcript_hf_event_strength": event_strength,
        })

    # Match the rest of the pipeline's convention: emit a daily calendar in [start, end)
    # and fill non-session days with neutral values.
    bar_dates = pd.date_range(start=start_dt, end=end_dt, freq="D", inclusive="left")
    bar_df = pd.DataFrame({"date": bar_dates})
    bar_df = bar_df.merge(session_df, on="date", how="left")
    fill_cols = [c for c in bar_df.columns if c != "date"]
    bar_df[fill_cols] = bar_df[fill_cols].fillna(0.0)
    return bar_df


def _create_empty_signal(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    """Create empty signal with neutral values."""
    bar_dates = pd.date_range(start=start_dt, end=end_dt, freq="D", inclusive="left")
    return pd.DataFrame({
        "date": bar_dates,
        "score": 0.0,
        "conf": 0.0,
        "has_data": 0.0,
    })


def _validate_signal(df: pd.DataFrame) -> None:
    """Validate signal format."""
    required_cols = {"date", "score", "conf"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Signal missing required columns: {missing}")
    
    if df["date"].isna().any():
        raise ValueError("Signal has NaN dates")
    
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Signal dates not sorted")
