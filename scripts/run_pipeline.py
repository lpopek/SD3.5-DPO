#!/usr/bin/env python3
"""
Orchestration: config -> preference pairs -> Diffusion-DPO -> LoRA.

    python scripts/run_pipeline.py --config configs/aesthetic.yaml --prompts prompts.txt
    python scripts/run_pipeline.py --config configs/technical.yaml --prompts prompts.txt
    # only pairs:            --skip-train
    # only training (pairs exist): --skip-generate
    # training on the shared-pool pairs (v2, scripts/build_pairs.py):
    #   --pairs /path/rl_aes/pairs_v2/preferences.jsonl --run-name lora_v2
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="override (e.g. 5 for a smoke test)")
    ap.add_argument("--beta", type=float, default=None, help="override beta")
    ap.add_argument("--lr", type=float, default=None, help="override learning_rate")
    ap.add_argument("--grad-accum", type=int, default=None, help="override grad_accum (effective batch)")
    ap.add_argument("--run-name", default="lora", help="subfolder of out_dir for this run (sweeps)")
    ap.add_argument("--init-lora", default=None,
                    help="continue training from this DPO adapter folder (e.g. .../rl_aes/lora_v2)")
    ap.add_argument("--pairs", default=None,
                    help="preferences.jsonl to train on (e.g. pairs_v2 from the shared pool); implies --skip-generate")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda"
    out_dir = cfg["out_dir"]
    pairs_dir = os.path.join(out_dir, "pairs")
    jsonl = args.pairs or os.path.join(pairs_dir, "preferences.jsonl")
    if args.pairs:
        args.skip_generate = True
    print(f"axis={cfg['axis']}  metrics={list(cfg['metrics'])}  out={out_dir}  pairs={jsonl}")

    if not args.skip_generate:
        if args.prompts:
            prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
        else:
            prompts = ["Ruf Guitar Schrodinger 6. An electric guitar with a black burst body "
                       "and a carbon-fiber top, glossy finish, studio product photograph."]
            print("WARNING: no --prompts, using one test prompt")

        from reward.metrics import RewardModel
        from data.generate import load_pipeline, build_dataset
        import torch

        rm = RewardModel(axis=cfg["axis"], device=device, weights=cfg["metrics"])
        pipe = load_pipeline(cfg["base_model"], cfg.get("plain_lora"), device)
        build_dataset(prompts, pipe, rm, pairs_dir,
                      k=cfg["k_per_prompt"], base_seed=cfg["base_seed"],
                      max_pairs=cfg["max_pairs_per_prompt"], margin=cfg["margin"],
                      steps=cfg.get("gen_steps", 28), guidance=cfg.get("gen_guidance", 7.0),
                      size=cfg.get("resolution", 1024))
        del pipe, rm
        import gc; gc.collect(); torch.cuda.empty_cache()

    if args.skip_train:
        print("--skip-train: stopping after pair generation"); return

    from train.dpo import DPOConfig, build_policy, train_dpo
    from train.data import PreferenceDataset

    dcfg = DPOConfig.from_dict(cfg)
    if args.max_steps:
        dcfg.max_steps = args.max_steps
    if args.beta is not None:
        dcfg.beta = args.beta
    if args.lr is not None:
        dcfg.learning_rate = args.lr
    if args.grad_accum is not None:
        dcfg.grad_accum = args.grad_accum
    print(f"DPO: beta={dcfg.beta} lr={dcfg.learning_rate} eff_batch={dcfg.batch_pairs*dcfg.grad_accum} "
          f"steps={dcfg.max_steps} -> {os.path.join(out_dir, args.run_name)}")
    ds = PreferenceDataset(jsonl, resolution=dcfg.resolution, min_margin=cfg.get("margin", 0.0))
    print("dataset:", ds.summary())
    pipe, transformer = build_policy(cfg["base_model"], cfg.get("plain_lora"), dcfg, device,
                                     init_lora=args.init_lora)
    train_dpo(pipe, transformer, ds, dcfg, os.path.join(out_dir, args.run_name), device=device)
    print(f"done. DPO LoRA: {os.path.join(out_dir, args.run_name)}")


if __name__ == "__main__":
    main()
