#!/usr/bin/env python3
"""
CPU test of the shared-pool path (no diffusers / pyiqa / GPU):

    python tests/test_pool_pairs.py

Fake pipeline (deterministic colour from seed + prompt) and fake reward models
(image statistics) exercise: image-level resume when K grows, legacy import by
prompt string, incremental scoring on both axes, the three pairing strategies,
and that the written preferences.jsonl loads with train/data.PreferenceDataset.
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import zlib

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate import build_pool, import_legacy_images, load_pool_meta  # noqa: E402
from reward.scoring import score_pool, load_scores  # noqa: E402
from scripts.build_pairs import pairs_for_axis, summarize, write_outputs  # noqa: E402
from scripts.build_selection import select, summarize as sel_summary, ARMS  # noqa: E402
from reward.metrics import composite_from_raw  # noqa: E402
from train.data import PreferenceDataset  # noqa: E402


class FakeOut:
    def __init__(self, im): self.images = [im]


class FakePipe:
    """Colour = hash(prompt, seed): same (prompt, seed) -> identical image."""
    calls = 0

    def __call__(self, prompt, num_inference_steps, guidance_scale, height, width, generator):
        FakePipe.calls += 1
        seed = generator.initial_seed()
        h = zlib.crc32(f"{prompt}|{seed}".encode())
        rgb = (h & 255, (h >> 8) & 255, (h >> 16) & 255)
        return FakeOut(Image.new("RGB", (width, height), rgb))


class FakeReward:
    def __init__(self, axis):
        self.axis = axis
        self.metrics = {"nima": None, "laion_aes": None} if axis == "aesthetic" \
            else {"musiq": None, "topiq_nr": None, "clipiqa": None}

    def raw_scores(self, images):
        m = images.mean(dim=(2, 3))            # (N,3) channel means in [0,1]
        if self.axis == "aesthetic":
            return {"nima": m[:, 0] * 10, "laion_aes": m[:, 1] * 10}
        return {"musiq": m[:, 2] * 100, "topiq_nr": m.mean(1), "clipiqa": (m[:, 0] + m[:, 2]) / 2}


def main():
    tmp = tempfile.mkdtemp(prefix="sd35dpo_pool_")
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp):
    prompts = [f"Ruf Guitars Schrödinger 6. prompt {i}" for i in range(5)]
    pipe = FakePipe()
    settings = dict(base_seed=1000, steps=2, guidance=1.0, size=16)

    # --- legacy run: K=8 for prompts in a DIFFERENT order, per-axis layout ---
    legacy = os.path.join(tmp, "rl_aes", "pairs")
    os.makedirs(os.path.join(legacy, "images"))
    legacy_prompts = prompts[::-1]
    with open(os.path.join(legacy, "scores.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["prompt_idx", "gen", "path", "prompt", "nima", "laion_aes", "reward"])
        for pi, p in enumerate(legacy_prompts):
            for j in range(8):
                rel = f"images/p{pi:03d}_g{j}.png"
                pipe(p, 2, 1.0, 16, 16, torch.Generator().manual_seed(1000 + j)).images[0] \
                    .save(os.path.join(legacy, rel))
                wr.writerow([pi, j, rel, p, 0, 0, 0])
    FakePipe.calls = 0

    # --- pool: import legacy (by prompt), then grow to K=16 ---
    pool = os.path.join(tmp, "pool")
    os.makedirs(pool)
    n = import_legacy_images(pool, prompts, [legacy], k=16, base_seed=1000)
    assert n == 40, n
    # legacy p000 was the LAST prompt of the new list -> must land in p004
    a = Image.open(os.path.join(legacy, "images/p000_g3.png")).getpixel((0, 0))
    b = Image.open(os.path.join(pool, "images/p004_g3.png")).getpixel((0, 0))
    assert a == b, "legacy import must match by prompt string, not by index"

    s = build_pool(prompts, pipe, pool, k=16, **settings, log_every=0)
    assert s["n_generated"] == 40 and FakePipe.calls == 40, s
    # imported images are byte-identical to what the pool would generate itself
    ref = pipe(prompts[4], 2, 1.0, 16, 16, torch.Generator().manual_seed(1003)).images[0].getpixel((0, 0))
    assert b == ref
    FakePipe.calls = 0
    s = build_pool(prompts, None, pool, k=16, **settings, log_every=0)   # nothing missing -> no pipe needed
    assert s["n_generated"] == 0 and FakePipe.calls == 0
    assert load_pool_meta(pool)["k"] == 16

    # pool settings are frozen
    try:
        build_pool(prompts, pipe, pool, k=16, base_seed=1, steps=2, guidance=1.0, size=16, log_every=0)
        raise AssertionError("base_seed change must be rejected")
    except ValueError:
        pass
    try:
        build_pool(prompts[:4], pipe, pool, k=16, **settings, log_every=0)
        raise AssertionError("prompt list change must be rejected")
    except ValueError:
        pass

    # --- scoring: incremental, both axes ---
    rms = {"aesthetic": FakeReward("aesthetic"), "technical": FakeReward("technical")}
    csv_path = score_pool(pool, rms, batch_size=7)
    rows, metrics = load_scores(csv_path)
    assert len(rows) == 80 and set(metrics) == {"nima", "laion_aes", "musiq", "topiq_nr", "clipiqa"}, metrics
    assert rows[0]["seed"] == "1000" and rows[15]["seed"] == "1015"
    # grow to K=20: only 20 new images generated, only 20 new rows scored
    build_pool(prompts, pipe, pool, k=20, **settings, log_every=0)
    before = {r["path"]: dict(r) for r in rows}
    rows2, _ = load_scores(score_pool(pool, rms, batch_size=7))
    assert len(rows2) == 100
    for r in rows2:
        if r["path"] in before:
            assert r == before[r["path"]], "cached rows must be untouched"

    # --- pairs per axis ---
    weights = {"nima": 1.0, "laion_aes": 1.0}
    pairs, scored = pairs_for_axis(rows2, weights, "top_bottom", margin=0.0, max_pairs=8, min_rank_gap=1)
    assert len(pairs) == 5 * 8, len(pairs)
    for p in pairs:
        assert p["chosen_reward"] > p["rejected_reward"]
        assert p["chosen"] != p["rejected"]
    pairs_k8, _ = pairs_for_axis(rows2, weights, "top_bottom", 0.0, 2, 1, k=8)
    assert len(pairs_k8) == 10 and all(p["chosen_gen"] < 8 and p["rejected_gen"] < 8 for p in pairs_k8)
    pairs_rg, _ = pairs_for_axis(rows2, weights, "ranked_gap", 0.0, 16, 4, k=16)
    assert len(pairs_rg) == 5 * 16
    pairs_all, _ = pairs_for_axis(rows2, weights, "all_above_margin", 0.0, 1000, 1, k=16)
    assert len(pairs_all) == 5 * 120
    big = pairs_for_axis(rows2, weights, "top_bottom", margin=1e9, max_pairs=8, min_rank_gap=1)[0]
    assert big == []

    # --- outputs load with PreferenceDataset (relative paths) ---
    out = os.path.join(tmp, "rl_aes", "pairs_v2")
    summary = summarize(pairs, scored, "top_bottom", 0.0, 8, 1)
    jsonl = write_outputs(pairs, scored, out, pool, weights, summary)
    ds = PreferenceDataset(jsonl, resolution=16)
    assert len(ds) == 40
    item = ds[0]
    assert item["chosen"].shape == (3, 16, 16) and item["rejected"].shape == (3, 16, 16)
    sm = json.load(open(os.path.join(out, "summary.json")))
    assert sm["n_pairs"] == 40 and sm["n_images_scored"] == 100 and sm["k_per_prompt"] == {"20": 5}
    print("summary:", json.dumps(sm)[:300])

    # --- reward-guided selection (variant A): 3 distinct images per prompt, best per axis
    import random
    prompt_to_ct = {p: (f"c{i}", "plain") for i, p in enumerate(prompts)}
    w_tech = {"musiq": 1.0, "topiq_nr": 1.0}
    picks, skipped = select(rows2, prompts, prompt_to_ct, weights, w_tech, random.Random(5))
    assert not skipped and len(picks) == 5
    for pk in picks:
        seeds = {pk[f"{a}_seed"] for a in ARMS}
        assert len(seeds) == 3, "three distinct seeds per prompt"
        g = [r for r in rows2 if int(r["prompt_idx"]) == pk["prompt_idx"]]
        raw = {m: torch.tensor([float(r[m]) for r in g]) for m in weights}
        best = int(g[int(torch.argmax(composite_from_raw(raw, weights)))]["seed"])
        assert pk["sel_aes_seed"] == best, "sel_aes must be the aesthetic argmax"
        assert pk["sel_aes_aes"] >= pk["random_aes"] and pk["sel_tech_tech"] >= pk["random_tech"]
    picks2, _ = select(rows2, prompts, prompt_to_ct, weights, w_tech, random.Random(5))
    assert picks2 == picks, "random arm must be reproducible for a given rng"
    ss = sel_summary(picks, {"20": 5}); assert ss["n_images"] == 15
    print("selection:", {k: ss[k] for k in ("n_prompts", "axes_agree_on_best", "gap_aes_mean", "gap_tech_mean")})
    print("OK — pool / legacy import / incremental scoring / pairs / dataset / selection")


if __name__ == "__main__":
    main()
