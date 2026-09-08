"""
Score every image of a shared pool with the raw metrics of BOTH reward axes
and cache the result in {pool_dir}/scores.csv:

    prompt_idx, gen, seed, path, prompt, nima, laion_aes, musiq, topiq_nr, clipiqa

Incremental: rows already present (all requested metric columns filled) are
kept, only new images are scored. Composite rewards are NOT stored here —
they are per axis and per prompt group, see scripts/build_pairs.py.
"""
from __future__ import annotations

import csv
import os

import torch
from PIL import Image

from data.generate import (POOL_IMAGES, load_pool_meta, load_pool_prompts,
                           pil_to_tensor, pool_image_paths)

SCORES_CSV = "scores.csv"
KEY_COLS = ["prompt_idx", "gen", "seed", "path", "prompt"]


def load_scores(path: str) -> tuple[list[dict], list[str]]:
    """rows (as dicts of str) + metric column names; ([], []) if missing."""
    if not os.path.exists(path):
        return [], []
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = list(rows[0].keys()) if rows else []
    return rows, [c for c in cols if c not in KEY_COLS]


def write_scores(path: str, rows: list[dict], metric_names: list[str]):
    rows = sorted(rows, key=lambda r: (int(r["prompt_idx"]), int(r["gen"])))
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=KEY_COLS + metric_names, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def _fmt(v) -> str:
    return f"{float(v):.5f}"


@torch.no_grad()
def score_pool(pool_dir: str, reward_models: dict, batch_size: int = 8,
               k: int | None = None) -> str:
    """
    reward_models: {axis_name: RewardModel-like} — each exposes `.metrics`
    (dict name -> callable) and `.raw_scores(images) -> {name: (N,)}`.
    Returns the path of scores.csv.
    """
    meta = load_pool_meta(pool_dir)
    if meta is None:
        raise FileNotFoundError(f"no meta.json in {pool_dir} — run build_pool first")
    prompts = load_pool_prompts(pool_dir)
    k = k or int(meta["k"])
    base_seed = int(meta["base_seed"])
    csv_path = os.path.join(pool_dir, SCORES_CSV)

    metric_names = [m for rm in reward_models.values() for m in rm.metrics]
    rows, old_metrics = load_scores(csv_path)
    by_path = {r["path"]: r for r in rows}

    grid = [(pi, j, rel) for pi, j, rel in pool_image_paths(pool_dir, len(prompts), k)
            if os.path.exists(os.path.join(pool_dir, rel))]
    n_missing_files = len(prompts) * k - len(grid)
    todo = [(pi, j, rel) for pi, j, rel in grid
            if rel not in by_path or any(not by_path[rel].get(m) for m in metric_names)]
    print(f"[score] pool {len(grid)} images on disk ({n_missing_files} of the grid missing), "
          f"{len(grid) - len(todo)} cached, {len(todo)} to score: {metric_names}")

    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        imgs = pil_to_tensor([Image.open(os.path.join(pool_dir, rel)) for _, _, rel in chunk])
        raw = {}
        for rm in reward_models.values():
            raw.update(rm.raw_scores(imgs))
        for n, (pi, j, rel) in enumerate(chunk):
            row = by_path.get(rel) or dict(prompt_idx=str(pi), gen=str(j), seed=str(base_seed + j),
                                           path=rel, prompt=prompts[pi])
            for m in metric_names:
                row[m] = _fmt(raw[m][n].item())
            by_path[rel] = row
        done = min(i + batch_size, len(todo))
        if done % (batch_size * 10) == 0 or done == len(todo):
            print(f"[score] {done}/{len(todo)}")
            write_scores(csv_path, list(by_path.values()), sorted(set(old_metrics) | set(metric_names)))

    write_scores(csv_path, list(by_path.values()), sorted(set(old_metrics) | set(metric_names)))
    print(f"[score] {len(by_path)} rows -> {csv_path}")
    return csv_path
