# Registry — Selection-Quality Conformance Battery

Recorded runs under [SPEC.md](SPEC.md) v0.1. Entries are
append-only; each names its runner, its kind, and its scope. An
entry records a run — not an endorsement, not production evidence.

## First-party reference run — 2026-10-26

**Runner:** Fame510 (first-party; the battery's author).
**Implementation commit:** recorded in each JSON entry.
**Track:** synthetic (SPEC.md §6). **Scope:** these numbers describe
behavior on the battery's pinned synthetic workloads in the
reference environment. They are not measurements of any real model —
for the real track, see `docs/real-attention-eval.md`, where the
magnitude signal below was tested against GPT-2's actual attention
and failed.

Mean over seeds 1–10. Mass recall / needle recall:

| Policy (mode) | Variant | @5% budget | @12% budget |
|---|---|---|---|
| singularity-selector (cold) | tell | 16.7% / 3.0% | 27.2% / 16.0% |
| singularity-selector (cold) | null | 16.7% / 3.0% | 27.2% / 16.0% |
| singularity-selector (warm) | tell | 25.6% / 15.0% | 59.8% / 60.0% |
| singularity-selector (warm) | null | 25.6% / 15.0% | 59.8% / 60.0% |
| singularity-selector+magnitude (cold) | tell | 23.8% / 25.0% | 58.0% / 70.0% |
| singularity-selector+magnitude (cold) | null | 14.1% / 2.0% | **22.5% / 11.0%** |
| heavy-hitter-observed (warm) | tell | 24.2% / 22.5% | 59.8% / 60.0% |
| heavy-hitter-observed (warm) | null | 24.2% / 22.5% | 59.8% / 60.0% |
| heavy-hitter-observed (cold) | both | *no_history_signal — recorded, not scored* | |

Entries: `registry/singularity-selector__cold__2026-10-26.json`,
`registry/singularity-selector__warm__2026-10-26.json`,
`registry/singularity-selector+magnitude__cold_magnitude__2026-10-26.json`,
`registry/heavy-hitter-observed__warm__2026-10-26.json`,
`registry/heavy-hitter-observed__cold__2026-10-26.json`.

### What this run establishes

- **Cold-start position selection is structurally needle-blind:**
  16% needle recall at the 12% budget. Any cold-start claim for a
  position-only policy must answer to this row.
- **The magnitude signal is construction-dependent, and the battery
  shows both directions:** with a key-magnitude tell it is the best
  cold policy in the table (70% needle recall); with no tell it is
  *worse than the position baseline it augments* (22.5% vs 27.2% mass
  recall) — it spends budget chasing norm noise. Together with its
  real-track failure, this is why it ships opt-in.
- **With real history, the shipped selector and a pure heavy-hitter
  converge** (59.8% / 60.0% at 12%) — observed attention dominates
  the blend, as designed.

## Independent runs

*None yet.* The battery is published so that this section can stop
being empty. Run it, file an entry with your scope statement, and
your numbers stand next to these on the same terms.
