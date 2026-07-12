"""Core data structures and algorithms for the SINGULARITY KV-cache fabric."""
from .blocks import BlockMeta, KVBlock, Tier
from .page_table import Entry, FetchPlan, GlobalPageTable
from .selector import SelectionConfig, assign_tiers, head_bytes, score_blocks, total_bytes

__all__ = [
    "BlockMeta", "KVBlock", "Tier",
    "Entry", "FetchPlan", "GlobalPageTable",
    "SelectionConfig", "assign_tiers", "head_bytes", "score_blocks", "total_bytes",
]
