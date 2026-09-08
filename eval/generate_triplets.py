#!/usr/bin/env python3
"""
Generate the survey images for one method (plain / rl_aes / rl_tech) from the
mos-eval catalogue (tasks_catalog.json), same prompt + seed per method, NEXT TO
the existing plain images:

    {out}/SD35/{color}/{top}/seed_{seed:04d}.png         plain   (already there)
    {out}/SD35/{color}/{top}/seed_{seed:04d}_aes.png     rl_aes
    {out}/SD35/{color}/{top}/seed_{seed:04d}_tech.png    rl_tech

    --out = .../Dataset/RufGen/generated  (the folder that contains SD35/)

With --no-suffix every method writes plain `seed_{seed:04d}.png`, i.e. the
catalogue / bucket layout with one folder per method (v2 dataset):

    {out}/SD35/{color}/{top}/seed_{seed:04d}.png   with --out .../generated_v2/{plain|generated_rl_aes|generated_rl_tech}

    python eval/generate_triplets.py --method rl_aes \
        --catalog /path/mos-eval/data/tasks_catalog.json \
        --base stabilityai/stable-diffusion-3.5-medium \
        --plain-lora /path/plain --dpo-lora /path/rl_aes/lora \
        --out /content/drive/MyDrive/rufgen --seeds 0-6

Then:  gsutil -m rsync -r {out}/SD35 gs://ruf-ai/rufgen/generated/SD35
Match --steps/--guidance to the settings used for the existing 'plain' images
(RufGen grid: --guidance 4.5 --steps 28, LoRA merged_finetune/rw). With
--manifest guitar_manifest.csv the prompt of each (color, top) is taken from the
manifest, i.e. the exact string the plain images were generated with; without it
the catalogue prompt is used with the token rewritten. --overwrite regenerates
files that already exist (default: skip them).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUFFIX = {"plain": "", "rl_aes": "_aes", "rl_tech": "_tech"}


def parse_seeds(s):
    if s is None:
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(SUFFIX), required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--base", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--plain-lora", required=True)
    ap.add_argument("--dpo-lora", default=None, help="required for rl_aes / rl_tech")
    ap.add_argument("--dpo-scale", type=float, default=1.0, help="DPO adapter strength at inference (1.0 = as trained)")
    ap.add_argument("--out", required=True, help="folder containing SD35/ (e.g. .../RufGen/generated)")
    ap.add_argument("--seeds", default=None, help="e.g. 0-6 or 5 (default: available_seeds)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--catalog-token", default="Ruf Guitars Schr\u00f6dinger 6")
    ap.add_argument("--train-token", default="Ruf Guitar Schrodinger 6",
                    help="trigger the plain LoRA was trained with; catalogue prompts are rewritten to it")
    ap.add_argument("--manifest", default=None, help="RufGen guitar_manifest.csv: prompt per (color, top)")
    ap.add_argument("--overwrite", action="store_true", help="regenerate existing files")
    ap.add_argument("--no-suffix", action="store_true",
                    help="write seed_XXXX.png for every method (one --out folder per method, catalogue layout)")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks (smoke test)")
    args = ap.parse_args()
    if args.method != "plain" and not args.dpo_lora:
        ap.error("--dpo-lora is required for rl_aes / rl_tech")

    import torch
    from train.dpo import load_policy_pipeline

    tasks = json.load(open(args.catalog, encoding="utf-8"))["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    lookup = None
    if args.manifest:
        from data.generate import manifest_prompt_lookup
        lookup = manifest_prompt_lookup(args.manifest)
        missing = [t["id"] for t in tasks if (t["meta"]["color"], t["meta"]["top"]) not in lookup]
        if missing:
            ap.error(f"{len(missing)} tasks without a manifest prompt, e.g. {missing[:3]}")
        print(f"[triplets] prompts from manifest ({len(lookup)} color/top combos)")
    pipe = load_policy_pipeline(args.base, args.plain_lora, args.dpo_lora, dpo_scale=args.dpo_scale)

    suffix = "" if args.no_suffix else SUFFIX[args.method]
    n_done = n_skip = 0
    for t in tasks:
        m = t["meta"]
        seeds = parse_seeds(args.seeds) or m.get("available_seeds", [m["seed"]])
        d = os.path.join(args.out, m["model"], m["color"], m["top"])
        os.makedirs(d, exist_ok=True)
        for s in seeds:
            p = os.path.join(d, f"seed_{s:04d}{suffix}.png")
            if os.path.exists(p) and not args.overwrite:
                n_skip += 1; continue
            g = torch.Generator(device="cpu").manual_seed(s)
            if lookup is not None:
                prompt = lookup[(m["color"], m["top"])]
            else:
                prompt = t["prompt"].replace(args.catalog_token, args.train_token)
            img = pipe(prompt=prompt, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, height=args.size, width=args.size,
                       generator=g).images[0]
            img.save(p); n_done += 1
        print(f"{t['id']}: ok")
    print(f"generated {n_done}, skipped {n_skip} existing -> {args.out} (suffix {suffix!r})")


if __name__ == "__main__":
    main()
