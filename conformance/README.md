# Selection-Quality Conformance Battery

Throughput benchmarks measure how much data moved. This battery
measures whether the **right** data moved.

- **Specification:** [SPEC.md](SPEC.md) — pinned workloads, adapter
  contract, metric definitions, registry rules.
- **Registry:** [REGISTRY.md](REGISTRY.md) — recorded runs, with
  first-party and independent runs labeled as what they are.
- **Reference runner:** `run_battery.py` — generates the pinned
  cases with the repository's own evaluator code and writes registry
  entries for the registered policies:

```bash
python3 conformance/run_battery.py
```

To record a run of your own implementation: implement the adapter
contract in SPEC.md §3 against the pinned generator, run the 10
seeds × 2 variants × 2 budgets, and add one JSON entry per
(policy, mode) under `registry/` with your scope statement. First
party or independent, the format is the same — the label is the
difference, and the label is mandatory.
