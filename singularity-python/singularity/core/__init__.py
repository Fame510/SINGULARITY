"""Core data structures and algorithms for the SINGULARITY KV-cache fabric."""
from .blocks import BlockMeta, KVBlock, Tier
from .codec import CODECS, Codec, EncodedTensor, Int8QuantCodec, RawCodec, quantization_error
from .eviction import LRUEvictionPolicy
from .metrics import Metrics
from .page_table import Entry, FetchPlan, GlobalPageTable
from .selector import SelectionConfig, assign_tiers, head_bytes, score_blocks, total_bytes

__all__ = [
    "BlockMeta", "KVBlock", "Tier",
    "CODECS", "Codec", "EncodedTensor", "Int8QuantCodec", "RawCodec", "quantization_error",
    "LRUEvictionPolicy", "Metrics",
    "Entry", "FetchPlan", "GlobalPageTable",
    "SelectionConfig", "assign_tiers", "head_bytes", "score_blocks", "total_bytes",
]
