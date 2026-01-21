"""
HF (Hugging Face) Generators
=============================

Modular, leak-safe generators for HF-based signals.
Each module follows a consistent interface and is registered in hf_registry.yaml.

Generator Contract:
    def build(
        symbol: str,
        horizon: int,
        start: str,         # ISO date (inclusive)
        end: str,           # ISO date (exclusive)
        out_path: str,      # Output parquet path
        raw_source_cfg: dict,  # Where to read raw data
        compute_cfg: dict,     # Model params, batch size, etc.
    ) -> None:
        '''
        Reads ONLY data within [start, end) for leak-safety,
        computes features aligned to bar calendar,
        and writes parquet with columns: date, score, conf
        '''

Available Generators:
    - news_sentiment_hf: News sentiment using FinBERT or similar
    - earnings_transcript_hf: Earnings call transcript analysis
    - doc_embedding_novelty_hf: Document embedding novelty detection
    - macro_tst_hf: Macroeconomic text signals transformer

Usage:
    from tools.hf_generators import get_generator
    
    generator = get_generator('news_sentiment_hf')
    generator.build(
        symbol='AAPL',
        horizon=63,
        start='2020-01-01',
        end='2020-12-31',
        out_path='data/local_cache/aapl_h63_news_sentiment_hf_train.parquet',
        raw_source_cfg={'news_dir': 'data/news'},
        compute_cfg={'model': 'ProsusAI/finbert', 'batch_size': 32}
    )
"""

import importlib
from pathlib import Path
from typing import Callable, Dict

import yaml


def get_registry_path() -> Path:
    """Get path to HF registry YAML."""
    return Path(__file__).parent.parent / "hf_registry.yaml"


def load_registry() -> Dict[str, Dict]:
    """Load HF generator registry from YAML."""
    registry_path = get_registry_path()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"HF registry not found: {registry_path}\n"
            "Run 'python tools/hf_generators/init_registry.py' to create it."
        )
    
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    
    return registry.get("generators", {})


def get_generator(module_name: str) -> Callable:
    """
    Get generator function for an HF module.
    
    Args:
        module_name: Name of HF module (e.g., 'news_sentiment_hf')
        
    Returns:
        Generator function with build() signature
        
    Raises:
        ValueError: If module not found in registry
        ImportError: If generator module cannot be imported
    """
    registry = load_registry()
    
    if module_name not in registry:
        available = ', '.join(registry.keys())
        raise ValueError(
            f"HF module '{module_name}' not found in registry.\n"
            f"Available modules: {available}"
        )
    
    config = registry[module_name]
    generator_path = config.get("generator")
    
    if not generator_path:
        raise ValueError(
            f"HF module '{module_name}' has no 'generator' field in registry"
        )
    
    # Import generator module dynamically
    # Format: "tools.hf_generators.news_sentiment_hf"
    try:
        module = importlib.import_module(generator_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import generator '{generator_path}' for module '{module_name}': {e}"
        )
    
    # Get build function
    if not hasattr(module, "build"):
        raise AttributeError(
            f"Generator module '{generator_path}' has no 'build' function"
        )
    
    return module


def list_generators() -> Dict[str, Dict]:
    """
    List all registered HF generators with their configs.
    
    Returns:
        Dict mapping module name to config dict
    """
    return load_registry()


def validate_generator(module_name: str) -> bool:
    """
    Validate that a generator can be loaded and has correct interface.
    
    Args:
        module_name: Name of HF module
        
    Returns:
        True if valid, False otherwise
    """
    try:
        generator = get_generator(module_name)
        
        # Check build function signature
        import inspect
        sig = inspect.signature(generator.build)
        required_params = {'symbol', 'horizon', 'start', 'end', 'out_path', 
                          'raw_source_cfg', 'compute_cfg'}
        actual_params = set(sig.parameters.keys())
        
        if not required_params.issubset(actual_params):
            missing = required_params - actual_params
            print(f"❌ Generator '{module_name}' missing parameters: {missing}")
            return False
        
        print(f"✅ Generator '{module_name}' validated")
        return True
        
    except Exception as e:
        print(f"❌ Generator '{module_name}' validation failed: {e}")
        return False
