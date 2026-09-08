#!/usr/bin/env python3
"""
Online RL (DDPO objective, Flow-GRPO SDE sampler) for SD-3.5 with metric rewards.

    python scripts/run_ddpo.py --config configs/ddpo_aesthetic.yaml --prompts prompts.txt
    python scripts/run_ddpo.py --config configs/ddpo_technical.yaml --manifest guitar_manifest.csv
    # resume:  --init-lora .../rl_aes/lora_ddpo --start-epoch 10
    # smoke:   --epochs 1 --prompts-per-epoch 1 --k 4

Prompts: --prompts file | --manifest guitar_manifest.csv | --catalog combinations.json
(same rules as scripts/build_pool.py). Output: {out_dir}/{run_name}/ (LoRA in diffusers
format + train_log.json) — loadable by eval/generate_triplets.py --dpo-lora.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--run-name", default="lora_ddpo")
    ap.add_argument("--init-lora", default=None, help="continue from a saved adapter")
    ap.add_argument("--start-epoch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--prompts-per-epoch", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="samples per prompt")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--beta-kl", type=float, default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--identity-lam", type=float, default=None, help="override identity_lam (0 = off)")
    ap.add_argument("--adv-mode", choices=["group", "global"], default=None)
    ap.add_argument("--shape-lam", type=float, default=None, help="override shape_lam (0 = off)")
    ap.add_argument("--shape-tol", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    from scripts.build_pool import load_prompts
    prompts = load_prompts(args, cfg.get("pool_dir", ""), cfg)

    from train.ddpo import DDPOConfig, train_ddpo
    from train.dpo import build_policy
    from reward.metrics import RewardModel

    dcfg = DDPOConfig.from_dict(cfg)
    for k, v in (("epochs", args.epochs), ("prompts_per_epoch", args.prompts_per_epoch), ("k_per_prompt", args.k),
                 ("learning_rate", args.lr), ("beta_kl", args.beta_kl), ("resolution", args.resolution),
                 ("start_epoch", args.start_epoch), ("identity_lam", args.identity_lam), ("adv_mode", args.adv_mode),
                 ("shape_lam", args.shape_lam), ("shape_tol", args.shape_tol)):
        if v is not None:
            setattr(dcfg, k, v)
    out = os.path.join(cfg["out_dir"], args.run_name)
    print(f"axis={cfg['axis']} metrics={list(cfg['metrics'])} prompts={len(prompts)} -> {out}")
    print(f"DDPO: T={dcfg.num_steps} guidance={dcfg.guidance} noise={dcfg.noise_level} res={dcfg.resolution} "
          f"K={dcfg.k_per_prompt} prompts/epoch={dcfg.prompts_per_epoch} epochs={dcfg.epochs} "
          f"lr={dcfg.learning_rate} clip={dcfg.clip_range} beta_kl={dcfg.beta_kl} "
          f"identity_lam={dcfg.identity_lam} shape_lam={dcfg.shape_lam} shape_tol={dcfg.shape_tol} adv_mode={dcfg.adv_mode}")

    rm = RewardModel(axis=cfg["axis"], device=args.device, weights=cfg["metrics"])

    idm = None
    idc = cfg.get("identity") or {}
    if dcfg.identity_lam > 0:
        from reward.identity import IdentityReward, list_images, load_images
        os.makedirs(out, exist_ok=True)
        idm = IdentityReward(idc["reference_dirs"], device=args.device, model=idc.get("model", "openai/clip-vit-large-patch14"),
                             topk=int(idc.get("topk", 5)), cache_path=os.path.join(cfg["out_dir"], "identity_refs.npz"))
        calib = os.path.join(out, "identity_calib.json")
        if os.path.exists(calib):
            idm.load_calibration(calib); print(f"[identity] calibration loaded: tau={idm.tau:.4f} sigma={idm.sigma:.4f}")
        else:
            paths = list_images([idc["calib_dir"]])[: int(idc.get("calib_images", 128))]
            if not paths:
                sys.exit(f"identity.calib_dir has no images: {idc['calib_dir']}")
            idm.calibrate(load_images(paths, size=512), path=calib)
        print(f"[identity] lam={dcfg.identity_lam} adv_mode={dcfg.adv_mode}")

    pipe, transformer = build_policy(cfg["base_model"], cfg.get("plain_lora"), dcfg.dpo_view(), args.device,
                                     init_lora=args.init_lora)
    train_ddpo(pipe, transformer, prompts, rm, dcfg, out, device=args.device, identity_model=idm)
    print(f"done. DDPO LoRA: {out}")


if __name__ == "__main__":
    main()
