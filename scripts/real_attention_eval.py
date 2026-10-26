#!/usr/bin/env python3
"""
SINGULARITY real-attention validation experiment.

Question: does the selector's magnitude signal — the mean L2 norm of a
block's key vectors — predict which KV blocks a REAL model actually
attends to, better than position alone?

Method (no synthetic ground truth anywhere in this script):
  1. Run a small real model (GPT-2, CPU) over real prose passages with
     attention capture on.
  2. GROUND TRUTH: the attention mass each token actually RECEIVES,
     summed over every layer, head, and query position. Aggregated to
     blocks of 16 tokens.
  3. SIGNALS scored per block:
       - magnitude: mean L2 norm of the block's key vectors, taken
         from the model's own KV cache (the exact quantity
         selector.magnitude_outlier_scores computes).
       - position: sink + recency scoring (the pre-patch selector's
         only cold-start signal).
  4. Metrics per passage: Spearman rank correlation of each signal
     against true block mass, and mass_recall at the 12% teleport
     budget — of the total attention mass, how much lands in the
     top-12% blocks each signal would teleport.

Whatever the numbers say is the result. Run:
    python real_attention_eval.py [--model openai-community/gpt2]
Prints JSON and a plain-language verdict.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

BLOCK = 16
BUDGET = 0.12  # the Sparse Teleportation head fraction

PASSAGES = [
    # Passage 1 — expository, with an early distinctive protocol that a
    # later paragraph refers back to.
    """The harbor at Bellmarsh runs on a tide protocol that nobody wrote
down until 1962. When the fog bell rings twice, the ferries hold at the
outer marker and wait for a green lamp from the signal house. The lamp
keeper checks the barometer, the tide gauge, and the chalk board where
the night watch records wind shifts in a hand nobody else can read. If
the board shows a falling glass and a northeast wind, the lamp stays
dark and the ferries ride at anchor until the bell rings three times.

The town grew around this arrangement without ever voting on it. Bakers
open when the first ferry lands. The school bell is set ten minutes
after the harbor bell, because children from the far side of the inlet
arrive on the early boat. Insurance men in the capital once tried to
replace the protocol with a printed timetable, and for one winter the
ferries ran on schedule instead of on signal. Three groundings later,
the town went back to the bell, the lamp, and the chalk board.

What makes the system durable is that every part of it is legible to
the people who depend on it. A child can read the lamp. A fisherman
can hear the bell through fog that hides everything else. The
barometer belongs to no one and argues with no one. Visitors sometimes
ask why the protocol described earlier — the two rings, the green
lamp, the chalk board with its falling glass — has never been written
into the town charter. The keepers answer that a charter can be
amended by people who were not out in the fog, and the bell cannot.""",
    # Passage 2 — technical explanation.
    """A heat pump moves warmth the way a refrigerator does, only in the
direction the householder wants. A refrigerant circulates through a
loop of copper, absorbing heat where it evaporates and surrendering it
where it condenses. The compressor does the work of squeezing the gas
until its temperature climbs above the temperature of the rooms, so
that heat flows downhill, from the hot coil to the cooler air, even
though the source of that heat was the colder garden outside.

The efficiency of the machine is not magic; it is bookkeeping. For
every unit of electricity the compressor consumes, the loop may
deliver three or four units of heat indoors, because most of the
delivered energy was collected outdoors rather than created. The
ratio falters when the outside air falls far below freezing, since
the refrigerant must then gather heat from air that holds very
little, and the compressor works harder for every degree of lift.

Installers argue about sizing the way sailors argue about anchors.
Too small a unit runs constantly and wears; too large a unit cycles
in short bursts and leaves the air clammy in summer. The careful
installer measures the house first — its insulation, its windows, the
leakage at its doors — and only then chooses the machine, because the
building, not the brochure, decides what the loop must carry across
the coldest week of the year.""",
    # Passage 3 — narrative.
    """Mara kept the lighthouse log in a hand so steady that the
inspectors used her pages to teach new keepers what a log should look
like. Wind, sea state, lamp trimmed at six, lamp lit at dusk: the
entries repeated for eleven years with the monotony of a metronome.
What the log never recorded was the garden she kept behind the tower,
in the lee of the wall, where she grew tomatoes at a latitude where
tomatoes were supposed to be impossible.

She had built the garden the way she kept the light, by patience and
by noticing. The wall stored the day's heat and returned it at night.
The path of white gravel threw light upward under the leaves. When
storms stripped the beds, she replanted without comment in the log,
because the log was for the shipping office, and the garden was hers.

In the last autumn of her service, a supply boat brought a young
assistant who asked, on his first evening, why the tower stores
listed twice the paraffin the lamp could burn. Mara showed him the
second ledger, the private one, where the extra oil was reckoned
against the nights the fog closed for a week and the relief boat
could not come. 'The log they read in the office,' she said, 'is the
light as they imagine it. This one is the light as it is.' She left
him both ledgers when she retired, and the garden, which he kept for
thirty years without ever writing it down either.""",
]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r

    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager"
    )
    model.eval()

    results = []
    for pi, text in enumerate(PASSAGES):
        ids = tok(text, return_tensors="pt").input_ids
        seq = ids.shape[1]
        with torch.no_grad():
            out = model(ids, output_attentions=True, use_cache=True)

        # ── ground truth: attention mass RECEIVED per token ──
        received = np.zeros(seq, dtype=np.float64)
        for layer_attn in out.attentions:  # [1, heads, q, kv]
            a = layer_attn[0].numpy().astype(np.float64)
            received += a.sum(axis=(0, 1))  # sum over heads and queries
        received /= len(out.attentions)

        # ── magnitude signal: per-token key norms from the KV cache ──
        cache = out.past_key_values
        key_caches = []
        if hasattr(cache, "key_cache"):  # transformers DynamicCache
            key_caches = list(cache.key_cache)
        elif hasattr(cache, "layers"):
            key_caches = [layer.keys for layer in cache.layers]
        else:  # legacy tuple of (k, v)
            key_caches = [kv[0] for kv in cache]
        key_norm = np.zeros(seq, dtype=np.float64)
        n_layers = len(key_caches)
        for k in key_caches:  # [1, heads, seq, head_dim]
            kn = np.linalg.norm(k[0].numpy().astype(np.float64), axis=-1)
            key_norm += kn.mean(axis=0)  # mean over heads
        key_norm /= n_layers

        # ── aggregate to blocks ──
        n_blocks = seq // BLOCK
        true_mass = np.array(
            [received[i * BLOCK:(i + 1) * BLOCK].sum() for i in range(n_blocks)]
        )
        magnitude = np.array(
            [key_norm[i * BLOCK:(i + 1) * BLOCK].mean() for i in range(n_blocks)]
        )
        # position signal: sink (block 0) + exponential recency of the
        # block's newest token — the pre-patch selector's cold start.
        position = np.array(
            [
                math.exp(-(seq - 1 - ((i + 1) * BLOCK - 1)) / 256.0)
                + (1.0 if i == 0 else 0.0)
                for i in range(n_blocks)
            ]
        )

        total_mass = true_mass.sum()
        k_top = max(1, int(round(BUDGET * n_blocks)))

        def recall_at_budget(signal: np.ndarray) -> float:
            top = np.argsort(signal)[::-1][:k_top]
            return float(true_mass[top].sum() / total_mass)

        oracle_recall = float(
            true_mass[np.argsort(true_mass)[::-1][:k_top]].sum() / total_mass
        )
        row = {
            "passage": pi,
            "tokens": int(seq),
            "blocks": int(n_blocks),
            "budget_blocks": int(k_top),
            "spearman_magnitude": spearman(magnitude, true_mass),
            "spearman_position": spearman(position, true_mass),
            "mass_recall_magnitude": recall_at_budget(magnitude),
            "mass_recall_position": recall_at_budget(position),
            "mass_recall_oracle_top": oracle_recall,
            "mass_recall_random_expected": BUDGET,
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    summary = {
        "model": args.model,
        "passages": len(results),
        "block_tokens": BLOCK,
        "budget": BUDGET,
        "mean_spearman_magnitude": mean("spearman_magnitude"),
        "mean_spearman_position": mean("spearman_position"),
        "mean_mass_recall_magnitude": mean("mass_recall_magnitude"),
        "mean_mass_recall_position": mean("mass_recall_position"),
        "mean_mass_recall_oracle_top": mean("mass_recall_oracle_top"),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    mag_wins = (
        summary["mean_spearman_magnitude"] > summary["mean_spearman_position"]
        and summary["mean_mass_recall_magnitude"]
        > summary["mean_mass_recall_position"]
    )
    print(
        "\nVERDICT: magnitude signal BEATS position on real attention traces"
        if mag_wins
        else "\nVERDICT: magnitude signal does NOT beat position on these traces"
    )
    with open("real_attention_results.json", "w") as f:
        json.dump({"summary": summary, "passages": results}, f, indent=2)


if __name__ == "__main__":
    main()
