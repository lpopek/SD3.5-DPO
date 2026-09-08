#!/usr/bin/env python3
"""
The metric that matters: does the DPO policy score higher on its reward axis
than the plain model, on HELD-OUT prompts and seeds?

For N prompts x S seeds it generates plain and policy images from the same
(prompt, seed), scores both with the axis reward (raw metrics, no within-batch
standardisation) and reports mean gain per metric + pairwise win-rate.

    python eval/reward_gain.py --config configs/aesthetic.yaml --dpo-lora .../rl_aes/lora \
        --prompts prompts.txt --n-prompts 20 --seeds 100-102 --out .../rl_aes/eval_lora
    # compare checkpoints / sweep runs by pointing --dpo-lora at each folder
"""
import argparse, csv, json, os, sys
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def gen(pipe, prompt, seed, steps, guidance, size):
    g = torch.Generator(device="cpu").manual_seed(seed)
    im = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
              height=size, width=size, generator=g).images[0]
    return im, torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float()[None] / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dpo-lora", required=True)
    ap.add_argument("--dpo-scale", type=float, default=1.0, help="DPO adapter strength at inference")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--seeds", default="100-101", help="disjoint from pair seeds (1000+) and survey seeds (0-19)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.0)
    args = ap.parse_args()

    from reward.metrics import RewardModel
    from train.dpo import load_policy_pipeline

    cfg = yaml.safe_load(open(args.config))
    size = cfg.get("resolution", 1024)
    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    rng = np.random.RandomState(0); rng.shuffle(prompts)
    prompts = prompts[:args.n_prompts]
    seeds = parse_seeds(args.seeds)
    os.makedirs(args.out, exist_ok=True)
    rm = RewardModel(axis=cfg["axis"], device="cuda", weights=cfg["metrics"])
    names = list(rm.metrics)

    rows = []
    for tag, dpo in (("plain", None), ("policy", args.dpo_lora)):
        pipe = load_policy_pipeline(cfg["base_model"], cfg.get("plain_lora"), dpo,
                                    dpo_scale=args.dpo_scale if dpo else 1.0)
        for pi, p in enumerate(prompts):
            for s in seeds:
                im, t = gen(pipe, p, s, args.steps, args.guidance, size)
                im.save(os.path.join(args.out, f"{tag}_p{pi:03d}_s{s}.png"))
                raw = rm.raw_scores(t)
                rows.append(dict(tag=tag, prompt_idx=pi, seed=s, **{n: raw[n].item() for n in names}))
        del pipe; torch.cuda.empty_cache()

    with open(os.path.join(args.out, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # paired comparison per (prompt, seed)
    key = lambda r: (r["prompt_idx"], r["seed"])
    plain = {key(r): r for r in rows if r["tag"] == "plain"}
    pol = {key(r): r for r in rows if r["tag"] == "policy"}
    report = {}
    for n in names:
        d = np.array([pol[k][n] - plain[k][n] for k in plain])
        report[n] = dict(mean_plain=float(np.mean([plain[k][n] for k in plain])),
                         mean_policy=float(np.mean([pol[k][n] for k in plain])),
                         mean_gain=float(d.mean()), win_rate=float((d > 0).mean()),
                         n=int(len(d)))
    # composite on the pooled set (same standardisation for both arms)
    allraw = {n: torch.tensor([r[n] for r in rows]) for n in names}
    comp = rm.composite_from_raw(allraw).numpy()
    tags = np.array([r["tag"] for r in rows])
    report["composite"] = dict(mean_plain=float(comp[tags == "plain"].mean()),
                               mean_policy=float(comp[tags == "policy"].mean()))
    report["_run"] = dict(dpo_lora=args.dpo_lora, dpo_scale=args.dpo_scale, n_prompts=len(prompts),
                          seeds=seeds, steps=args.steps, guidance=args.guidance)
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
