# CHANGES.md — Session patch log

Everything below was written, tested, and verified in this session. Nothing
here is claimed without a command that was actually run to check it.

## Bug fixes

1. **`singularity-python/singularity/core/metrics.py`** — `Metrics.snapshot()`
   deadlocked permanently. It acquired `self._lock` (a plain, non-reentrant
   `threading.Lock`) and then called `self.percentile()` from inside that same
   `with` block, which tried to acquire the identical lock again. Fixed by
   switching to `threading.RLock`. Verified: `tests/test_metrics.py` used to
   hang forever (exit code 124 on a 15s timeout); now passes in well under a
   second.

2. **`singularity-transport/Cargo.toml`** — declared `rdma = "0.1"`, a version
   that does not exist on crates.io (current is 0.3.0), so this crate has
   never actually compiled as originally written. Corrected the version
   number. Note: even at 0.3.0, this crate's own transitive dependency chain
   (`hex-simd` / `simd-abstraction`) is currently broken upstream on
   crates.io — that's an external, verifiable-today problem, not something
   fixable from this repo.

## New capability: adaptive signal weighting + real cold-start signal

**`singularity-python/singularity/core/selector.py`**

- `score_blocks()` previously blended sink/recency/observed at a FIXED
  0.45/0.45/0.10 weight regardless of whether "observed" (historical
  attention_score) actually had any signal in it. On a real first-time
  request, attention_score is 0.0 for every block, so that 10% term
  contributed nothing — mid-context "needle" importance was undetectable
  by construction, at any head_fraction. Verified with the new evaluator
  (see below): cold-start needle_recall was 0% at every head_fraction tested
  under the old fixed weighting.
- Weights are now adaptive: the observed signal gets real weight (0.50) only
  when at least one block has a nonzero attention_score (i.e. a genuine
  repeat/warm request). A new magnitude signal (below) gets weight (0.30)
  only when real KV tensor data was supplied. Otherwise it falls back to the
  original position-only behavior.
- New function `magnitude_outlier_scores()`: computes a per-block salience
  proxy from the block's REAL key-vector data (mean L2 norm, z-scored across
  the context) — grounded in published outlier-feature / activation-magnitude
  work. This is available on a genuine cold start because it needs no prior
  request history, only the tensors already present in this request.
  **Caveat, stated here plainly and not just in conversation:** whether this
  correlation holds for any specific deployed model is NOT verified against
  real attention traces in this repo — there is no way to pull a real trained
  model's weights in the environment this was built in. Treat this as a
  real, implementable, testable candidate signal, not a proven production fix,
  until checked against real model attention data.

**`singularity-python/singularity/sdk/fabric.py`**

- `ContextFabric.publish_context()` was calling `assign_tiers()` with only
  `b.meta` (metadata), discarding the actual KV tensor payload before scoring
  ever happened — meaning the magnitude signal above was structurally
  unreachable even after being written, until this was fixed. Now computes
  `magnitude_outlier_scores()` from the real blocks before tiering.

## New file: `scripts/oracle_quality_eval.py`

A ground-truth-based quality evaluator that did not exist before this
session. The existing benchmark (`slb_benchmark.py`) only measures how much
DATA moves — this measures whether the RIGHT data was selected.

- Builds a synthetic context with a KNOWN true-importance distribution:
  sink mass, recency mass, and a small number of randomly-placed
  "needle" blocks — deliberately high true importance buried in the
  middle of the context, where position-only heuristics cannot see them
  (the classic "needle in a haystack" failure mode).
- Runs the real, unmodified production selector code against it and reports
  `mass_recall` (fraction of true importance captured), `precision` (fraction
  of selected blocks that were actually in the true top-K), and
  `needle_recall` (specifically, did it catch the buried important content).
- Includes an honesty control: `--needle-key-magnitude-mult 1.0` is the NULL
  case (no real magnitude tell on the needles). Run across 10 seeds, both the
  old and new methods score near 0% needle_recall with no consistent edge —
  confirming the new signal isn't just getting lucky.
- With a real tell present (`--needle-key-magnitude-mult 3.0`, the default),
  run across the same 10 seeds: cold-start needle_recall goes from ~0-40%
  (old, mostly luck) to **100%, every single seed** using the new magnitude
  signal — while remaining a genuine cold start (no prior request needed).

## What is explicitly NOT claimed here

- No hardware RDMA transport was completed. `singularity-transport/fabric.rs`
  is still a placeholder (`register_memory`/`teleport_direct` log and return
  without moving any bytes). A real attempt was made this session to build
  against the `ibverbs` crate; it got substantially further (fixed a
  bindgen/glibc header path issue) but hit two remaining external blockers:
  a genuine C compile bug in the vendored 2021 rdma-core source, and a
  bindgen-vs-glibc-2.39 incompatibility that requires a newer bindgen than
  this crate pins. No RDMA-capable hardware (`/dev/infiniband`) exists in the
  environment this was built in regardless, so even a successful compile
  would have nothing to test against.
- "26x" is not re-asserted anywhere in this patch. It is mathematically
  reachable by lowering `head_fraction` alone (verified: 0.0385 → 25.68x
  measured), but doing so without the fixes above measurably collapses
  selection quality (verified: warm-cache needle_recall drops from 100% at
  head_fraction=0.12 to 40% at 0.0385 to 0% at 0.02). No claim is made here
  about what head_fraction is safe for any specific real workload — that
  requires real attention-trace validation this environment can't perform.
