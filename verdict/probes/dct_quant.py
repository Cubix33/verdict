"""Backwards-compatible shim. The real implementation lives in compression.py."""
from .compression import dct_quant, extract_qtable_hash  # noqa: F401
