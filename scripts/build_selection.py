#!/usr/bin/env python3
"""
Reward-guided SELECTION survey set (no training): for every prompt of the shared
pool pick, out of its K scored samples,

    random    one seed drawn at random (the "unguided" arm)
    sel_aes   the seed with the highest aesthetic composite   (NIMA + LAION-aes)
    sel_tech  the seed with the highest technical composite   (MUSIQ + TOPIQ-NR)

Three DIFFERENT images per prompt: the random seed is drawn from the seeds not
chosen by either selector; if both axes pick the same seed the technical arm
takes its runner-up (flagged in selection.csv, column tech_fallback).

Output (catalogue layout, one folder per arm, file name = pool seed):
    {out}/{random|sel_aes|sel_tech}/SD35/{color}/{top}/seed_{seed:04d}.png
    {out}/selection.csv          prompt_idx, color, top, per-arm seed/gen/composites, fallback flag
    {out}/selection_summary.json counts, agreement between axes, score gaps

    python scripts/build_selection.py --pool .../RufGen/pool --manifest .../guitar_manifest.csv \
        --out .../RufGen/generated_v2_sel --rng 5
Composites are the same robust-z-within-prompt rewards as in scripts/build_pairs.py.
"""
import argparse
import collections
import csv
import json
import os
import random
import shutil
import statistics
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate import load_pool_meta, load_pool_prompts, read_manifest  # noqa: E402
from reward.metrics import AXES, composite_from_raw  # noqa: E402
from reward.scoring import SCORES_CSV, load_scores  # noqa: E402

ARMS = ("random", "sel_aes", "sel_tech")


def axis_weights(config_path: str | None, axis: str) -> dict:
    if config_path:
        return {m: float(w) for m, w in yaml.safe_load(open(config_path))["metrics"].items()}
    return dict(AXES[axis])


def select(rows, prompts, prompt_to_ct, w_aes, w_tech, rng, k=None):
    groups = collections.defaultdict(list)
    for r in rows:
        if all(r.get(m) not in (None, "") for m in list(w_aes) + list(w_tech)):
            groups[int(r["prompt_idx"])].append(r)
    picks, skipped = [], []
    for pi in sorted(groups):
        g = sorted(groups[pi], key=lambda r: int(r["gen"]))
        if k is not None:
            g = [r for r in g if int(r["gen"]) < k]
        if len(g) < 3:
            skipped.append(pi); continue
        ct = prompt_to_ct.get(prompts[pi])
        if ct is None:
            skipped.append(pi); continue
        raw_a = {m: torch.tensor([float(r[m]) for r in g]) for m in w_aes}
        raw_t = {m: torch.tensor([float(r[m]) for r in g]) for m in w_tech}
        ca = composite_from_raw(raw_a, w_aes).tolist()
        ctc = composite_from_raw(raw_t, w_tech).tolist()
        order_a = sorted(range(len(g)), key=lambda i: -ca[i])
        order_t = sorted(range(len(g)), key=lambda i: -ctc[i])
        ia = order_a[0]
        it, fallback = order_t[0], False
        if it == ia:
            it, fallback = order_t[1], True
        rest = [i for i in range(len(g)) if i not in (ia, it)]
        ir = rng.choice(rest)
        rec = dict(prompt_idx=pi, color=ct[0], top=ct[1], prompt=prompts[pi], tech_fallback=int(fallback),
                   rank_random_aes=order_a.index(ir) + 1, rank_random_tech=order_t.index(ir) + 1)
        for arm, i in (("random", ir), ("sel_aes", ia), ("sel_tech", it)):
            r = g[i]
            rec[f"{arm}_gen"] = int(r["gen"]); rec[f"{arm}_seed"] = int(r["seed"]); rec[f"{arm}_path"] = r["path"]
            rec[f"{arm}_aes"] = round(ca[i], 5); rec[f"{arm}_tech"] = round(ctc[i], 5)
        picks.append(rec)
    return picks, skipped


def summarize(picks, k_hist):
    n = len(picks)
    gaps_a = [p["sel_aes_aes"] - p["random_aes"] for p in picks]
    gaps_t = [p["sel_tech_tech"] - p["random_tech"] for p in picks]
    same = sum(p["tech_fallback"] for p in picks)
    return dict(
        n_prompts=n, n_images=3 * n, k_per_prompt=k_hist,
        axes_agree_on_best=same, axes_agree_frac=round(same / n, 3) if n else None,
        gap_aes_mean=round(statistics.mean(gaps_a), 3) if n else None,
        gap_tech_mean=round(statistics.mean(gaps_t), 3) if n else None,
        random_rank_aes_mean=round(statistics.mean(p["rank_random_aes"] for p in picks), 2) if n else None,
        random_rank_tech_mean=round(statistics.mean(p["rank_random_tech"] for p in picks), 2) if n else None,
        seed_hist={arm: dict(collections.Counter(p[f"{arm}_seed"] for p in picks)) for arm in ARMS},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--manifest", required=True, help="guitar_manifest.csv (prompt -> color/top)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--aes-config", default=None, help="configs/aesthetic.yaml (metric weights)")
    ap.add_argument("--tech-config", default=None, help="configs/technical.yaml (metric weights)")
    ap.add_argument("--rng", type=int, default=5, help="seed of the random arm")
    ap.add_argument("--k", type=int, default=None, help="use only gens < k")
    ap.add_argument("--model", default="SD35")
    ap.add_argument("--no-copy", action="store_true", help="only write selection.csv / summary")
    args = ap.parse_args()

    meta = load_pool_meta(args.pool)
    if meta is None:
        sys.exit(f"no meta.json in {args.pool}")
    prompts = load_pool_prompts(args.pool)
    rows, present = load_scores(os.path.join(args.pool, SCORES_CSV))
    w_aes = axis_weights(args.aes_config, "aesthetic")
    w_tech = axis_weights(args.tech_config, "technical")
    missing = [m for m in list(w_aes) + list(w_tech) if m not in present]
    if missing:
        sys.exit(f"pool scores.csv lacks {missing} — run scripts/build_pool.py")
    prompt_to_ct = {}
    for r in read_manifest(args.manifest):
        prompt_to_ct.setdefault(r["prompt"], (r["color"], r["top"]))

    picks, skipped = select(rows, prompts, prompt_to_ct, w_aes, w_tech, random.Random(args.rng), k=args.k)
    if skipped:
        print(f"WARNING: {len(skipped)} prompts skipped (unscored / <3 samples / not in manifest): {skipped[:5]}")
    k_hist = dict(collections.Counter(collections.Counter(int(r["prompt_idx"]) for r in rows).values()))

    os.makedirs(args.out, exist_ok=True)
    n_copied = 0
    if not args.no_copy:
        for p in picks:
            for arm in ARMS:
                d = os.path.join(args.out, arm, args.model, p["color"], p["top"])
                os.makedirs(d, exist_ok=True)
                dst = os.path.join(d, f"seed_{p[f'{arm}_seed']:04d}.png")
                if not os.path.exists(dst):
                    shutil.copyfile(os.path.join(args.pool, p[f"{arm}_path"]), dst); n_copied += 1
    cols = list(picks[0].keys()) if picks else []
    with open(os.path.join(args.out, "selection.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(picks)
    summary = dict(summarize(picks, k_hist), rng=args.rng, pool=args.pool, metrics_aes=w_aes, metrics_tech=w_tech,
                   n_copied=n_copied, layout=f"{args.out}/{{random|sel_aes|sel_tech}}/{args.model}/{{color}}/{{top}}/seed_XXXX.png")
    json.dump(summary, open(os.path.join(args.out, "selection_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "seed_hist"}, indent=1))


if __name__ == "__main__":
    main()
