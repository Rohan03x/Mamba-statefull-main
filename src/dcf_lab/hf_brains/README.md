# Hugging Face Brain Modules

This package contains a small collection of Hugging Face powered signal modules
that emit `ModuleSignal` objects compatible with the DCF Suite aggregation
pipeline. Each module is implemented defensively: if the required dependencies
are missing the module will fall back to simpler heuristics instead of raising
runtime errors.

## Available modules

| Name | Description | Default horizons | Refresh cadence |
| ---- | ----------- | ---------------- | --------------- |
| `news_sentiment_hf` | FinBERT-based daily headline sentiment scoring. | 1, 5, 30 | Daily |
| `earnings_transcript_hf` | Quarterly earnings call tone/confidence summary. | 5, 30, 63 | Quarterly |
| `doc_embedding_novelty_hf` | Narrative novelty using sentence-transformer embeddings. | 1, 5, 30 | Daily |
| `macro_tst_hf` | Macro panel direction using TimeSeriesTransformer logits. | 5, 21, 63 | Weekly |

All modules are described in `HF_BRAIN_MANIFEST` (`hf_brains/__init__.py`) which
also exposes helper utilities to load individual modules or register them with
the universal aggregator:

```python
from src.dcf_lab.hf_brains import register_with_aggregator

# Register every HF brain module with the aggregator in one shot.
register_with_aggregator()
```

## Installation

Install the required Hugging Face stack (GPU-enabled) inside your active
environment:

```bash
pip install "transformers>=4.42" datasets accelerate sentence-transformers bitsandbytes
```

## GPU hygiene

On RTX cards under WSL/Linux, set the allocator split size before launching
`auto_opt` to avoid fragmentation issues:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

This value is applied automatically by `auto_opt` if it is not already set.

## Caching & model revisions

* The tooling automatically points Hugging Face caches at
  `~/.cache/hf_brains` (via `HF_HOME` / `TRANSFORMERS_CACHE`). Override the
  location with `--hf-cache-dir` or by exporting the variables yourself when you
  want to place caches on a faster volume.
* All modules pin their models to the `main` revision by default. Pass
  `model_revision="<commit-hash>"` to individual constructors if you need
  stricter reproducibility or to test unreleased branches.

## Runtime configuration

Several environment variables (and new `auto_opt` flags) control Hugging Face
runtime behaviour:

| Variable / flag | Purpose |
| --------------- | ------- |
| `--hf-device` / `HF_BRAINS_DEVICE` | Preferred inference device (`cpu`, `cuda:0`, `auto`). |
| `--hf-cache-dir` / `HF_BRAINS_CACHE_DIR` | Explicit cache directory for models/tokenizers. |
| `--hf-offline` / `HF_BRAINS_OFFLINE` | Force offline mode (no network downloads). |
| `--hf-hub-token` / `HF_TOKEN` | Authentication token for private models. |

Additional knobs (with sensible defaults) are exposed via module constructor
arguments: lookback window, maximum article count, novelty window size, and
model overrides.

## Dependency notes

* The modules prefer `transformers` ≥ 4.35 and (optionally) `torch` ≥ 2.1 for
  GPU acceleration.
* When `transformers` is not available the modules fall back to classic
  heuristics (e.g. VADER sentiment or volume-based novelty).
* All pipelines use safe defaults (`trust_remote_code=False`, safetensors
  cache) and reuse caches between calls.

Feel free to extend `HF_BRAIN_MANIFEST` with additional modules following the
same pattern—no changes to the aggregator or calibrator layers are required.
