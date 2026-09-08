"""
Preference-pair generation for Diffusion-DPO.

Two entry points:

1. build_pool()   — SHARED image pool (v2, scripts/build_pool.py)
       {pool_dir}/images/p{prompt_idx:03d}_g{k}.png   seed = base_seed + k
       {pool_dir}/prompts.txt                         prompt of row p{idx}
       {pool_dir}/meta.json                           k, base_seed, sampler settings
   Only MISSING images are generated, so the pool can be grown from K=8 to
   K=16 without touching the first 8 (same seeds). Scoring on both reward
   axes and pairing per axis live in reward/scoring.py + scripts/build_pairs.py.
   import_legacy_images() copies the images of the earlier per-axis runs
   ({out_dir}/pairs/images of rl_aes / rl_tech) into the pool, matched by the
   PROMPT STRING of their scores.csv, so index order does not matter.

2. build_dataset() — legacy single-axis path (scripts/run_pipeline.py):
       generate K -> score on one axis -> top-vs-bottom pairs -> preferences.jsonl
"""
from __future__ import annotations

import csv
import json
import os
import shutil

import numpy as np
import torch

POOL_META = "meta.json"
POOL_PROMPTS = "prompts.txt"
POOL_IMAGES = "images"
_POOL_FIXED = ("base_seed", "steps", "guidance", "size")


def load_pipeline(base_model: str, lora_path: str | None, device: str = "cuda"):
    """SD-3.5 pipeline with the plain LoRA fused (same as the DPO reference)."""
    from train.dpo import load_policy_pipeline
    return load_policy_pipeline(base_model, lora_path, None, device=device)


def read_manifest(path: str) -> list[dict]:
    """RufGen tables/guitar_manifest.csv rows (color, top, seed, prompt, ...)."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prompts_from_manifest(path: str) -> list[str]:
    """The unique prompt strings of the RufGen grid, sorted (178 for the 2026-08 grid)."""
    return sorted({r["prompt"] for r in read_manifest(path)})


def manifest_prompt_lookup(path: str) -> dict[tuple[str, str], str]:
    """{(color, top): prompt} — the exact string used for the survey's plain images."""
    out = {}
    for r in read_manifest(path):
        out.setdefault((r["color"], r["top"]), r["prompt"])
    return out


def image_name(prompt_idx: int, gen: int) -> str:
    return f"p{prompt_idx:03d}_g{gen}.png"


def pil_to_tensor(imgs) -> torch.Tensor:
    """list[PIL] -> (N,3,H,W) float in [0,1]."""
    return torch.stack([torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
                        for im in imgs])


def generate_one(pipe, prompt: str, seed: int, steps: int = 28, guidance: float = 7.0,
                 size: int = 1024):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                height=size, width=size, generator=g).images[0]


@torch.no_grad()
def generate_k(pipe, prompt: str, k: int, base_seed: int = 0,
               steps: int = 28, guidance: float = 7.0, size: int = 1024):
    """K images for one prompt with seeds base_seed..base_seed+k-1.
    Returns (list[PIL], tensor (K,3,H,W) in [0,1])."""
    imgs = [generate_one(pipe, prompt, base_seed + i, steps, guidance, size) for i in range(k)]
    return imgs, pil_to_tensor(imgs)


# ----------------------------------------------------------------------------
# shared pool (v2)
# ----------------------------------------------------------------------------
def load_pool_meta(pool_dir: str) -> dict | None:
    p = os.path.join(pool_dir, POOL_META)
    return json.load(open(p)) if os.path.exists(p) else None


def load_pool_prompts(pool_dir: str) -> list[str]:
    p = os.path.join(pool_dir, POOL_PROMPTS)
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]


def pool_image_paths(pool_dir: str, n_prompts: int, k: int) -> list[tuple[int, int, str]]:
    """[(prompt_idx, gen, relative path)] for the full grid."""
    return [(pi, j, os.path.join(POOL_IMAGES, image_name(pi, j)))
            for pi in range(n_prompts) for j in range(k)]


def _check_pool_meta(pool_dir: str, prompts: list[str], k: int, settings: dict) -> dict:
    """Merge the requested settings with an existing meta.json (or create it).
    Sampler settings and base_seed are frozen once the pool exists; K may grow."""
    meta = load_pool_meta(pool_dir)
    if meta is None:
        meta = dict(k=k, n_prompts=len(prompts), **settings)
        return meta
    for key in _POOL_FIXED:
        if meta.get(key) != settings[key]:
            raise ValueError(f"pool {pool_dir} was built with {key}={meta.get(key)}, "
                             f"now {settings[key]} — change the pool_dir instead")
    old = load_pool_prompts(pool_dir)
    if old != prompts:
        n_diff = sum(1 for a, b in zip(old, prompts) if a != b) + abs(len(old) - len(prompts))
        raise ValueError(f"pool {pool_dir} has different prompts ({n_diff} differ / "
                         f"{len(old)} vs {len(prompts)}); a pool is tied to one prompt list")
    meta["k"] = max(int(meta["k"]), k)
    return meta


def _write_pool_meta(pool_dir: str, prompts: list[str], meta: dict):
    with open(os.path.join(pool_dir, POOL_PROMPTS), "w", encoding="utf-8") as f:
        f.write("\n".join(prompts) + "\n")
    with open(os.path.join(pool_dir, POOL_META), "w") as f:
        json.dump(meta, f, indent=1)


def import_legacy_images(pool_dir: str, prompts: list[str], legacy_dirs: list[str],
                         k: int, base_seed: int) -> int:
    """
    Copy p{idx}_g{gen}.png of earlier single-axis runs into the pool.
    A legacy dir is a `pairs/` folder with scores.csv (prompt_idx, gen, path, prompt).
    Matching is by PROMPT STRING; the legacy gen index must be < k. The legacy
    runs used seeds base_seed+gen with the SAME base_seed (configs: 1000) —
    this is asserted against a `meta.json` next to the legacy scores if present,
    otherwise trusted. Returns the number of files copied.
    """
    os.makedirs(os.path.join(pool_dir, POOL_IMAGES), exist_ok=True)
    idx = {p: i for i, p in enumerate(prompts)}
    copied = 0
    for ld in legacy_dirs:
        sc = os.path.join(ld, "scores.csv")
        if not os.path.exists(sc):
            print(f"[pool] legacy dir without scores.csv, skipped: {ld}")
            continue
        lmeta = load_pool_meta(ld)
        if lmeta and lmeta.get("base_seed") != base_seed:
            raise ValueError(f"legacy {ld} has base_seed={lmeta.get('base_seed')} != {base_seed}")
        n_here = n_unknown = 0
        for row in csv.DictReader(open(sc, encoding="utf-8")):
            pi = idx.get(row["prompt"])
            gen = int(row["gen"])
            if pi is None:
                n_unknown += 1
                continue
            if gen >= k:
                continue
            src = os.path.join(ld, row["path"])
            dst = os.path.join(pool_dir, POOL_IMAGES, image_name(pi, gen))
            if os.path.exists(dst) or not os.path.exists(src):
                continue
            shutil.copyfile(src, dst)
            copied += 1; n_here += 1
        print(f"[pool] {ld}: copied {n_here} images"
              + (f", {n_unknown} rows with prompts not in the list" if n_unknown else ""))
    return copied


def build_pool(prompts: list[str], pipe, pool_dir: str, k: int = 16, base_seed: int = 1000,
               steps: int = 28, guidance: float = 7.0, size: int = 1024,
               limit: int | None = None, log_every: int = 1) -> dict:
    """
    Generate the images missing from the pool grid (prompts x k). `pipe` may be
    None when nothing is missing (e.g. scoring-only runs). Returns a summary dict.
    """
    os.makedirs(os.path.join(pool_dir, POOL_IMAGES), exist_ok=True)
    settings = dict(base_seed=base_seed, steps=steps, guidance=guidance, size=size)
    meta = _check_pool_meta(pool_dir, prompts, k, settings)
    _write_pool_meta(pool_dir, prompts, meta)

    todo = [(pi, j, rel) for pi, j, rel in pool_image_paths(pool_dir, len(prompts), k)
            if not os.path.exists(os.path.join(pool_dir, rel))]
    if limit is not None:
        todo = [t for t in todo if t[0] < limit]
    n_total = len(prompts) * k
    print(f"[pool] grid {len(prompts)} prompts x K={k} = {n_total} images; "
          f"{n_total - len(todo)} present, {len(todo)} to generate -> {pool_dir}")
    if todo and pipe is None:
        raise RuntimeError(f"{len(todo)} images missing but no pipeline given")

    n_done = 0
    for n, (pi, j, rel) in enumerate(todo, 1):
        im = generate_one(pipe, prompts[pi], base_seed + j, steps, guidance, size)
        im.save(os.path.join(pool_dir, rel))
        n_done += 1
        if log_every and (n % log_every == 0 or n == len(todo)):
            print(f"[pool] {n}/{len(todo)} {rel}  (prompt {pi}: {prompts[pi][:45]}...)")
    return dict(pool_dir=pool_dir, n_prompts=len(prompts), k=k, n_total=n_total,
                n_generated=n_done, n_present=n_total - len(todo) + n_done)


# ----------------------------------------------------------------------------
# legacy single-axis dataset (kept for scripts/run_pipeline.py)
# ----------------------------------------------------------------------------
def build_dataset(prompts, pipe, reward_model, out_dir, k=8, base_seed=0,
                  max_pairs=2, margin=0.1, steps=28, guidance=7.0, size=1024,
                  resume=True):
    """generate -> score -> pair -> write. Returns path to preferences.jsonl."""
    from reward.pairs import build_preference_pairs

    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "preferences.jsonl")
    csv_path = os.path.join(out_dir, "scores.csv")

    metric_names = list(reward_model.metrics.keys())
    n_pairs = 0
    with open(jsonl_path, "w") as fj, open(csv_path, "w", newline="") as fc:
        wr = csv.writer(fc)
        wr.writerow(["prompt_idx", "gen", "path", "prompt", *metric_names, "reward"])
        for pi, prompt in enumerate(prompts):
            paths = [os.path.join("images", image_name(pi, j)) for j in range(k)]
            have_all = resume and all(os.path.exists(os.path.join(out_dir, p)) for p in paths)
            if have_all:
                from PIL import Image
                pil = [Image.open(os.path.join(out_dir, p)).convert("RGB") for p in paths]
                tens = pil_to_tensor(pil)
            else:
                pil, tens = generate_k(pipe, prompt, k, base_seed, steps, guidance, size)
                for im, p in zip(pil, paths):
                    im.save(os.path.join(out_dir, p))

            raw = reward_model.raw_scores(tens)          # {metric: (K,)}
            rewards = reward_model.composite_from_raw(raw)  # (K,)
            for j in range(k):
                wr.writerow([pi, j, paths[j], prompt,
                             *[f"{raw[m][j].item():.5f}" for m in metric_names],
                             f"{rewards[j].item():.5f}"])

            pairs = build_preference_pairs(tens, rewards, prompt, strategy="top_bottom",
                                           margin=margin, max_pairs=max_pairs)
            for pr in pairs:
                fj.write(json.dumps({
                    "prompt": prompt,
                    "chosen": paths[pr["chosen_idx"]],
                    "rejected": paths[pr["rejected_idx"]],
                    "chosen_reward": pr["chosen_reward"],
                    "rejected_reward": pr["rejected_reward"],
                }) + "\n")
            n_pairs += len(pairs)
            print(f"[{pi+1}/{len(prompts)}] {prompt[:50]}... -> {len(pairs)} pairs "
                  f"(reward range {rewards.min():.2f}..{rewards.max():.2f})")
    print(f"\n{n_pairs} pairs written to {jsonl_path}; per-image scores in {csv_path}")
    return jsonl_path
