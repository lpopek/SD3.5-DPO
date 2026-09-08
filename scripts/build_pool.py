#!/usr/bin/env python3
"""
Shared preference pool (v2): grow the K samples per prompt and score them on
BOTH reward axes in one pass. Pairs per axis: scripts/build_pairs.py.

    python scripts/build_pool.py --config configs/pool.yaml --catalog combinations.json
    # prompts, in order of precedence: --prompts file.txt | --manifest guitar_manifest.csv
    #          (the exact RufGen grid prompts, 178 unique, sorted) | --catalog combinations.json
    #          (catalogue prompts rewritten to the training token) | config `manifest` |
    #          existing {pool_dir}/prompts.txt
    # scoring only (images complete):      --skip-generate
    # generation only (score later):       --skip-score
    # smoke test on the first N prompts:   --limit 3

Resumable at image level: only missing p{idx}_g{k}.png are generated, only
unscored images are scored. The first K=8 of every prompt are imported from
the earlier per-axis runs (config `legacy_image_dirs`), matched by prompt.
"""
import argparse
import gc
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_prompts(args, pool_dir, cfg):
    """--prompts > --manifest > --catalog > config manifest > existing {pool_dir}/prompts.txt."""
    import json
    if args.prompts:
        return [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    manifest = args.manifest or (None if args.catalog else cfg.get("manifest"))
    if manifest:
        from data.generate import prompts_from_manifest
        prompts = prompts_from_manifest(manifest)
        print(f"[pool] {len(prompts)} unique prompts from manifest {manifest}")
        return prompts
    if args.catalog:
        tasks = json.load(open(args.catalog, encoding="utf-8"))["tasks"]
        prompts = sorted({t["prompt"].replace(args.catalog_token, args.train_token) for t in tasks})
        bad = [p for p in prompts if args.train_token not in p]
        if bad:
            print(f"WARNING: {len(bad)} prompts without the training token, e.g. {bad[0][:60]}")
        print(f"[pool] {len(prompts)} prompts from {args.catalog} (token {args.train_token!r})")
        return prompts
    existing = os.path.join(pool_dir, "prompts.txt")
    if os.path.exists(existing):
        print(f"[pool] prompts from existing pool: {existing}")
        return [l.rstrip("\n") for l in open(existing, encoding="utf-8") if l.strip()]
    sys.exit("no prompts: pass --catalog combinations.json or --prompts file.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line (pool order)")
    ap.add_argument("--manifest", default=None, help="RufGen guitar_manifest.csv -> its unique prompt strings")
    ap.add_argument("--catalog", default=None, help="combinations.json / tasks_catalog.json -> prompts")
    ap.add_argument("--catalog-token", default="Ruf Guitars Schr\u00f6dinger 6")
    ap.add_argument("--train-token", default="Ruf Guitar Schrodinger 6")
    ap.add_argument("--k", type=int, default=None, help="override k_per_prompt")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--no-legacy", action="store_true", help="do not import legacy images")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke test)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pool_dir = cfg["pool_dir"]
    k = args.k or int(cfg["k_per_prompt"])
    base_seed = int(cfg["base_seed"])
    prompts = load_prompts(args, pool_dir, cfg)
    print(f"pool={pool_dir}  prompts={len(prompts)}  K={k}  seeds={base_seed}..{base_seed + k - 1}")

    from data.generate import build_pool, import_legacy_images, load_pipeline
    os.makedirs(pool_dir, exist_ok=True)

    if not args.no_legacy and cfg.get("legacy_image_dirs"):
        n = import_legacy_images(pool_dir, prompts, cfg["legacy_image_dirs"], k, base_seed)
        print(f"[pool] legacy images imported: {n}")

    if not args.skip_generate:
        # load SD-3.5 lazily: only if something is missing
        from data.generate import pool_image_paths
        missing = [rel for pi, _, rel in pool_image_paths(pool_dir, len(prompts), k)
                   if (args.limit is None or pi < args.limit)
                   and not os.path.exists(os.path.join(pool_dir, rel))]
        pipe = None
        if missing:
            pipe = load_pipeline(cfg["base_model"], cfg.get("plain_lora"), args.device)
        summary = build_pool(prompts, pipe, pool_dir, k=k, base_seed=base_seed,
                             steps=cfg.get("gen_steps", 28), guidance=cfg.get("gen_guidance", 7.0),
                             size=cfg.get("resolution", 1024), limit=args.limit)
        print("[pool]", summary)
        if pipe is not None:
            del pipe
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

    if not args.skip_score:
        from reward.metrics import RewardModel
        from reward.scoring import score_pool
        rms = {axis: RewardModel(axis=axis, device=args.device, weights=w)
               for axis, w in cfg["axes"].items()}
        score_pool(pool_dir, rms, batch_size=int(cfg.get("score_batch_size", 8)), k=k)


if __name__ == "__main__":
    main()
