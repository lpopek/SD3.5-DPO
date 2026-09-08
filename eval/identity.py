#!/usr/bin/env python3
"""
Identity of generated images vs the reference photographs (CLIP ViT-L/14, top-k
cosine similarity) — for triplet folders and DDPO checkpoints / rollouts.

    python eval/identity.py --refs .../gitarki/1024/whole .../gitarki/1024/body \
        --dirs .../generated_v2/plain .../generated_v2/generated_rl_aes .../generated_v2/generated_rl_tech \
        --out identity_report.csv [--calib .../lora_ddpo/identity_calib.json] [--limit 200]

Per folder: mean / std identity; per image a CSV row (folder, relative path, identity,
term). Images with the same relative path across folders (same color/top/seed) are
compared pairwise against the FIRST folder (delta column), so "how much identity did
the adapter cost, same prompt and seed" is read directly.
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward.identity import IdentityReward, list_images, load_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="image folders; the first one is the baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=None, help="identity_calib.json of a DDPO run (for the term column)")
    ap.add_argument("--limit", type=int, default=None, help="first N images per folder (sorted)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    idm = IdentityReward(args.refs, device=args.device, topk=args.topk)
    if args.calib:
        idm.load_calibration(args.calib)

    rows, per_dir = [], {}
    for d in args.dirs:
        paths = list_images([d])[: args.limit] if args.limit else list_images([d])
        scores = []
        for i in range(0, len(paths), args.batch):
            scores += idm.score(load_images(paths[i:i + args.batch], size=512)).tolist()
        per_dir[d] = {os.path.relpath(p, d): s for p, s in zip(paths, scores)}
        print(f"{d}: n={len(scores)} identity mean={statistics.mean(scores):.4f} sd={statistics.pstdev(scores):.4f}")
    base = per_dir[args.dirs[0]]
    if idm.tau is None:
        idm.tau, idm.sigma = statistics.mean(base.values()), max(statistics.pstdev(base.values()), 1e-3)
        print(f"(term relative to the first folder: tau={idm.tau:.4f} sigma={idm.sigma:.4f})")
    import torch
    summary = {}
    for d, sc in per_dir.items():
        deltas = [sc[k] - base[k] for k in sc if k in base and d != args.dirs[0]]
        terms = idm.term(torch.tensor(list(sc.values()))).tolist()
        summary[d] = dict(n=len(sc), identity_mean=statistics.mean(sc.values()), identity_sd=statistics.pstdev(sc.values()),
                          term_mean=statistics.mean(terms),
                          delta_vs_first_mean=statistics.mean(deltas) if deltas else 0.0,
                          frac_below_first=(sum(x < 0 for x in deltas) / len(deltas)) if deltas else None)
        for (k, s), t in zip(sc.items(), terms):
            rows.append(dict(folder=d, path=k, identity=round(s, 5), term=round(t, 4),
                             delta_vs_first=round(s - base[k], 5) if k in base else ""))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump(dict(summary=summary, tau=idm.tau, sigma=idm.sigma, refs=args.refs),
              open(os.path.splitext(args.out)[0] + "_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
