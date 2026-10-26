#!/usr/bin/env python3
"""
SINGULARITY Selection-Quality Conformance Battery — reference runner.

Implements conformance/SPEC.md v0.1. Generates the pinned synthetic
workloads (via the exact generator in scripts/oracle_quality_eval.py,
so fixtures are identical to the evaluator's), runs each registered
policy over every case, and writes one registry JSON per policy into
conformance/registry/.

Usage:
    python3 conformance/run_battery.py [--out conformance/registry]

The policies registered here are the ones with entries in the
registry. External implementations follow the adapter contract in
SPEC.md and file their own entries.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "singularity-python"))

import numpy as np  # noqa: E402

from singularity.core.selector import SelectionConfig  # noqa: E402

# Load the evaluator module by path: its build_synthetic_context IS the
# normative workload generator for this battery (SPEC.md §2).
_spec = importlib.util.spec_from_file_location(
    "oracle_quality_eval", REPO / "scripts" / "oracle_quality_eval.py"
)
oqe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oqe)

# ── Pinned workload (SPEC.md §2) ─────────────────────────────────────
CONTEXT_TOKENS = 32768
BLOCK_TOKENS = 256
SINK_TOKENS = 32
RECENCY_TOKENS = 512
NUM_NEEDLES = 20
NEEDLE_MASS_EACH = 0.8
SEEDS = list(range(1, 11))
HEAD_FRACTIONS = [0.05, 0.12]
VARIANTS = {"tell": 3.0, "null": 1.0}  # needle key-magnitude multiplier


def metrics_for_hot(hot_idxs, true_importance, needle_idxs):
    """The battery's metric definitions (SPEC.md §4) — identical
    formulas to oracle_quality_eval.evaluate."""
    n = len(true_importance)
    n_hot = len(hot_idxs)
    ranked = sorted(range(n), key=lambda i: true_importance[i], reverse=True)
    true_top_k = set(ranked[:n_hot])
    mass_recall = sum(true_importance[i] for i in hot_idxs) / sum(true_importance)
    precision = len(hot_idxs & true_top_k) / n_hot if n_hot else 0.0
    needle_recall = (
        len(hot_idxs & needle_idxs) / len(needle_idxs) if needle_idxs else None
    )
    return {
        "n_hot_selected": n_hot,
        "mass_recall": round(mass_recall, 4),
        "precision_vs_true_top_k": round(precision, 4),
        "needle_recall": round(needle_recall, 4) if needle_recall is not None else None,
    }


def heavy_hitter_case(metas, true_importance, needle_idxs, n_hot):
    """H2O-class adapter: rank purely by observed attention mass and
    take the top-K. The observation is the same noisy one the warm
    selector path receives (Random(99), sigma = 0.15 * max mass), so
    the comparison is signal-for-signal. A heavy-hitter policy has no
    cold-start signal; the cold case is the caller's problem (this
    function is only invoked for warm cases)."""
    rng = random.Random(99)
    scores = []
    peak = max(true_importance)
    for t in true_importance:
        scores.append(max(0.0, t + rng.gauss(0, 0.15 * peak)))
    hot = set(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_hot])
    return metrics_for_hot(hot, true_importance, needle_idxs)


POLICIES = [
    {
        "id": "singularity-selector",
        "mode": "cold",
        "description": "The shipped SINGULARITY selector (singularity.core.selector) "
                       "on a true cold start: sink + recency position signals only.",
        "signals": ["sink", "recency"],
    },
    {
        "id": "singularity-selector",
        "mode": "warm",
        "description": "The shipped selector with a noisy prior-observation "
                       "attention history (repeat-request path).",
        "signals": ["sink", "recency", "observed_attention"],
    },
    {
        "id": "singularity-selector+magnitude",
        "mode": "cold_magnitude",
        "description": "The shipped selector plus the opt-in key-magnitude "
                       "signal. SYNTHETIC TRACK ONLY — this signal failed "
                       "real-attention validation (docs/real-attention-eval.md).",
        "signals": ["sink", "recency", "key_magnitude"],
    },
    {
        "id": "heavy-hitter-observed",
        "mode": "warm",
        "description": "Independent H2O-class adapter: top-K purely by observed "
                       "attention mass. No position signal, no cold-start path.",
        "signals": ["observed_attention"],
    },
]


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "conformance" / "registry"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    workload = {
        "context_tokens": CONTEXT_TOKENS,
        "block_tokens": BLOCK_TOKENS,
        "sink_tokens": SINK_TOKENS,
        "recency_tokens": RECENCY_TOKENS,
        "num_needles": NUM_NEEDLES,
        "needle_mass_each": NEEDLE_MASS_EACH,
        "seeds": SEEDS,
        "head_fractions": HEAD_FRACTIONS,
        "variants": VARIANTS,
    }

    # cases[(policy_key, variant, hf)] = list of per-seed case rows
    collected: dict = {}
    for variant, mult in VARIANTS.items():
        for seed in SEEDS:
            metas, true_imp, needle_idxs, kv_blocks = oqe.build_synthetic_context(
                CONTEXT_TOKENS, BLOCK_TOKENS, SINK_TOKENS, RECENCY_TOKENS,
                NUM_NEEDLES, NEEDLE_MASS_EACH, seed,
                needle_key_magnitude_mult=mult,
            )
            n_hot_by_hf = {}
            for hf in HEAD_FRACTIONS:
                cfg = SelectionConfig(
                    head_fraction=hf, sink_tokens=SINK_TOKENS,
                    recency_tokens=RECENCY_TOKENS,
                )
                for pol in POLICIES:
                    key = (pol["id"], pol["mode"], variant, hf)
                    if pol["id"] == "heavy-hitter-observed":
                        row = heavy_hitter_case(
                            metas, true_imp, needle_idxs,
                            n_hot_by_hf[hf],
                        )
                        row = {"seed": seed, "status": "ok", **row}
                    else:
                        res = oqe.evaluate(
                            metas, true_imp, needle_idxs, kv_blocks, cfg,
                            CONTEXT_TOKENS, mode=pol["mode"],
                        )
                        if pol["mode"] == "cold":
                            n_hot_by_hf[hf] = res["n_hot_selected"]
                        row = {"seed": seed, "status": "ok", **res}
                    collected.setdefault(key, []).append(row)
            # cold-start rows for the warm-only policy: recorded, not run
            for hf in HEAD_FRACTIONS:
                key = ("heavy-hitter-observed", "cold", variant, hf)
                collected.setdefault(key, []).append({
                    "seed": seed, "status": "no_history_signal",
                    "n_hot_selected": None, "mass_recall": None,
                    "precision_vs_true_top_k": None, "needle_recall": None,
                })

    written = []
    for pol in POLICIES:
        for mode in {pol["mode"]} | ({"cold"} if pol["id"] == "heavy-hitter-observed" else set()):
            cases, aggregates = [], []
            for variant in VARIANTS:
                for hf in HEAD_FRACTIONS:
                    rows = collected.get((pol["id"], mode, variant, hf), [])
                    for r in rows:
                        cases.append({"variant": variant, "head_fraction": hf, **r})
                    ok = [r for r in rows if r["status"] == "ok"]
                    if ok:
                        aggregates.append({
                            "variant": variant, "head_fraction": hf,
                            "cases": len(ok),
                            "mean_mass_recall": round(float(np.mean([r["mass_recall"] for r in ok])), 4),
                            "mean_precision_vs_true_top_k": round(float(np.mean([r["precision_vs_true_top_k"] for r in ok])), 4),
                            "mean_needle_recall": round(float(np.mean([r["needle_recall"] for r in ok])), 4),
                        })
            entry = {
                "battery": "singularity-selection-quality",
                "battery_version": "0.1",
                "track": "synthetic",
                "recorded": "2026-10-26",
                "runner": {
                    "name": "Fame510",
                    "kind": "first-party",
                    "scope": "Reference-environment run by the battery's author. "
                             "Synthetic track only; not an independent reproduction "
                             "and not evidence about any real model's attention.",
                },
                "implementation": {
                    "name": pol["id"],
                    "repo": "Fame510/SINGULARITY",
                    "commit": git_head(),
                },
                "policy": {
                    "id": pol["id"], "mode": mode,
                    "description": pol["description"],
                    "signals": pol["signals"],
                },
                "environment": env,
                "workload": workload,
                "cases": cases,
                "aggregates": aggregates,
            }
            fname = f"{pol['id']}__{mode}__2026-10-26.json"
            (out_dir / fname).write_text(json.dumps(entry, indent=2))
            written.append(fname)
            agg = "; ".join(
                f"{a['variant']}@{a['head_fraction']}: mass {a['mean_mass_recall']:.1%} "
                f"needle {a['mean_needle_recall']:.1%}"
                for a in aggregates
            )
            print(f"{pol['id']} [{mode}]  {agg}")

    print(f"\nwrote {len(written)} registry entries to {out_dir}")


if __name__ == "__main__":
    main()
