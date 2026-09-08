#!/usr/bin/env python3
"""
Preference pairs for ONE axis from the shared pool's scores.csv (no GPU, no pyiqa).

    python scripts/build_pairs.py --config configs/aesthetic.yaml
    python scripts/build_pairs.py --config configs/technical.yaml
    # overrides: --strategy ranked_gap --max-pairs 16 --margin 0.1 --min-rank-gap 4

Reads the `pairs_v2` block of the axis config (pool_dir, strategy, ...), the
metric weights of the axis, and writes

    {pairs_v2.out_dir | out_dir/out_subdir}/preferences.jsonl   (paths relative to that folder)
    {out_dir}/{pairs_v2.out_subdir}/scores.csv          per-image composite + z per metric
    {out_dir}/{pairs_v2.out_subdir}/summary.json        counts, margins, coverage

Composite reward = weighted mean of robust-z, standardised WITHIN the K samples
of each prompt (same as the legacy path, now over K=16 instead of 8).
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward.metrics import composite_from_raw, robust_z  # noqa: E402
from reward.pairs import STRATEGIES, build_preference_pairs  # noqa: E402
from reward.scoring import SCORES_CSV, load_scores  # noqa: E402


def pairs_for_axis(rows: list[dict], weights: dict, strategy: str, margin: float,
                   max_pairs: int, min_rank_gap: int, k: int | None = None):
    """
    rows: scores.csv rows of the pool. Returns (pairs, scored_rows) where a pair
    has pool-relative chosen/rejected paths and scored_rows carries the composite
    reward and per-metric z of every image used.
    """
    groups = collections.defaultdict(list)
    for r in rows:
        if all(r.get(m) not in (None, "") for m in weights):
            groups[int(r["prompt_idx"])].append(r)
    pairs, scored = [], []
    degenerate = collections.Counter()   # metric -> prompts whose MAD is ~0 (robust-z explodes)
    for pi in sorted(groups):
        g = sorted(groups[pi], key=lambda r: int(r["gen"]))
        if k is not None:
            g = [r for r in g if int(r["gen"]) < k]
        if len(g) < 2:
            continue
        raw = {m: torch.tensor([float(r[m]) for r in g]) for m in weights}
        rewards = composite_from_raw(raw, weights)
        zs = {m: robust_z(raw[m]) for m in weights}
        for m in weights:
            if (raw[m] - raw[m].median()).abs().median() < 1e-6:
                degenerate[m] += 1
        for n, r in enumerate(g):
            scored.append({**r, "reward": f"{rewards[n].item():.5f}",
                           **{f"z_{m}": f"{zs[m][n].item():.5f}" for m in weights}})
        for pr in build_preference_pairs(len(g), rewards, g[0]["prompt"], strategy=strategy,
                                         margin=margin, max_pairs=max_pairs,
                                         min_rank_gap=min_rank_gap):
            pairs.append(dict(prompt=pr["prompt"], prompt_idx=pi,
                              chosen=g[pr["chosen_idx"]]["path"], rejected=g[pr["rejected_idx"]]["path"],
                              chosen_gen=int(g[pr["chosen_idx"]]["gen"]),
                              rejected_gen=int(g[pr["rejected_idx"]]["gen"]),
                              chosen_reward=pr["chosen_reward"], rejected_reward=pr["rejected_reward"]))
    pairs_for_axis.degenerate_mad = dict(degenerate)
    return pairs, scored


def summarize(pairs, scored, strategy, margin, max_pairs, min_rank_gap) -> dict:
    per_prompt = collections.Counter(p["prompt_idx"] for p in pairs)
    margins = [p["chosen_reward"] - p["rejected_reward"] for p in pairs]
    used = {p["chosen"] for p in pairs} | {p["rejected"] for p in pairs}
    n_prompts = len({int(r["prompt_idx"]) for r in scored})
    k_hist = collections.Counter(collections.Counter(int(r["prompt_idx"]) for r in scored).values())
    return dict(
        strategy=strategy, margin=margin, max_pairs_per_prompt=max_pairs, min_rank_gap=min_rank_gap,
        n_images_scored=len(scored), n_prompts=n_prompts, k_per_prompt={str(k): v for k, v in sorted(k_hist.items())},
        n_pairs=len(pairs), n_images_in_pairs=len(used),
        prompts_with_pairs=len(per_prompt),
        pairs_per_prompt={str(k): v for k, v in sorted(collections.Counter(per_prompt.values()).items())},
        margin_mean=statistics.mean(margins) if margins else None,
        margin_median=statistics.median(margins) if margins else None,
        margin_min=min(margins) if margins else None, margin_max=max(margins) if margins else None,
        # prompts where a metric is constant over its K samples: its robust-z is then
        # +-huge and dominates the composite (CLIP-IQA saturates at ~1.0 on this data)
        degenerate_mad_prompts=getattr(pairs_for_axis, "degenerate_mad", {}),
    )


def write_outputs(pairs, scored, out_dir, pool_dir, weights, summary):
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "preferences.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "prompt": p["prompt"],
                "chosen": os.path.relpath(os.path.join(pool_dir, p["chosen"]), out_dir),
                "rejected": os.path.relpath(os.path.join(pool_dir, p["rejected"]), out_dir),
                "chosen_reward": p["chosen_reward"], "rejected_reward": p["rejected_reward"],
                "prompt_idx": p["prompt_idx"], "chosen_gen": p["chosen_gen"], "rejected_gen": p["rejected_gen"],
            }, ensure_ascii=False) + "\n")
    if scored:
        cols = list(scored[0].keys())
        with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader(); wr.writerows(scored)
    json.dump(dict(summary, pool_dir=pool_dir, metrics=weights),
              open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="axis config with a pairs_v2 block")
    ap.add_argument("--pool", default=None, help="override pairs_v2.pool_dir")
    ap.add_argument("--out", default=None, help="override output folder")
    ap.add_argument("--strategy", choices=STRATEGIES, default=None)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--min-rank-gap", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="use only gens < k (ablation)")
    ap.add_argument("--min-pairs", type=int, default=None, help="warn when fewer pairs than this")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pv = dict(cfg.get("pairs_v2", {}))
    pool_dir = args.pool or pv["pool_dir"]
    out_dir = args.out or pv.get("out_dir") or os.path.join(cfg["out_dir"], pv.get("out_subdir", "pairs_v2"))
    strategy = args.strategy or pv.get("strategy", "top_bottom")
    margin = args.margin if args.margin is not None else float(pv.get("margin", 0.1))
    max_pairs = args.max_pairs or int(pv.get("max_pairs_per_prompt", 8))
    min_rank_gap = args.min_rank_gap or int(pv.get("min_rank_gap", 1))
    weights = {m: float(w) for m, w in cfg["metrics"].items()}

    rows, present = load_scores(os.path.join(pool_dir, SCORES_CSV))
    missing = [m for m in weights if m not in present]
    if missing:
        sys.exit(f"pool scores.csv lacks metrics {missing} — run scripts/build_pool.py first")
    pairs, scored = pairs_for_axis(rows, weights, strategy, margin, max_pairs, min_rank_gap, k=args.k)
    summary = summarize(pairs, scored, strategy, margin, max_pairs, min_rank_gap)
    jsonl = write_outputs(pairs, scored, out_dir, pool_dir, weights, summary)
    print(f"axis={cfg['axis']} metrics={list(weights)}")
    print(json.dumps(summary, indent=1))
    print(f"-> {jsonl}")
    if args.min_pairs and summary["n_pairs"] < args.min_pairs:
        print(f"WARNING: {summary['n_pairs']} pairs < target {args.min_pairs}; "
              f"raise K, max_pairs, or use --strategy ranked_gap")


if __name__ == "__main__":
    main()
