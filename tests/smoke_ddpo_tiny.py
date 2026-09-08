#!/usr/bin/env python3
"""
CPU test of the DDPO / Flow-GRPO loop on a TINY random SD3 transformer (no downloads):

    python tests/smoke_ddpo_tiny.py

Checks: (1) the SDE step + log-prob are self-consistent (recomputing log-probs with
the unchanged policy gives ratio == 1), (2) PPO on a synthetic reward (mean of the
final latent) raises that reward over a few epochs, (3) the adapter saves/loads.
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel  # noqa: E402
from diffusers.training_utils import cast_training_params  # noqa: E402
from peft import LoraConfig  # noqa: E402

from train.ddpo import (DDPOConfig, cfg_velocity, gaussian_logp, sample_trajectories,  # noqa: E402
                        sde_mean_std, train_ddpo)
from train.dpo import SD3_LORA_TARGET_MODULES, load_lora_into_adapter  # noqa: E402

C, H, heads, hd, L, P = 16, 8, 2, 8, 6, 24


def tiny_transformer():
    tr = SD3Transformer2DModel(
        sample_size=H, patch_size=2, in_channels=C, out_channels=C, num_layers=2,
        attention_head_dim=hd, num_attention_heads=heads,
        joint_attention_dim=32, caption_projection_dim=heads * hd,
        pooled_projection_dim=P, pos_embed_max_size=16,
    )
    tr.requires_grad_(False)
    tr.add_adapter(LoraConfig(r=4, lora_alpha=8, init_lora_weights="gaussian",
                              target_modules=SD3_LORA_TARGET_MODULES))
    cast_training_params(tr, dtype=torch.float32)
    return tr


class FakePipe:
    """Just enough of StableDiffusion3Pipeline for train_ddpo on CPU."""
    def __init__(self, transformer):
        self.transformer = transformer
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        self.device = torch.device("cpu")
        self.text_encoder = self.text_encoder_2 = self.text_encoder_3 = object()
        self.vae = None

    def encode_prompt(self, prompt, **kw):
        g = torch.Generator().manual_seed(abs(hash(tuple(prompt))) % (2 ** 31))
        pe = torch.randn(len(prompt), L, 32, generator=g)
        po = torch.randn(len(prompt), P, generator=g)
        return pe, None, po, None


def main():
    torch.manual_seed(0)
    tr = tiny_transformer()
    pipe = FakePipe(tr)
    cfg = DDPOConfig(num_steps=4, guidance=2.0, noise_level=0.7, resolution=H * 8, k_per_prompt=4,
                     prompts_per_epoch=2, sample_batch=4, epochs=1, train_batch=4, grad_accum=1,
                     learning_rate=0.0, clip_range=1e-4, gradient_checkpointing=False, seed=0)

    # --- (1) log-prob consistency: sample, then recompute logp with the same weights
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    pe, _, po, _ = pipe.encode_prompt(prompt=["a"]); pn, _, pon, _ = pipe.encode_prompt(prompt=[""])
    traj = sample_trajectories(tr, scheduler, cfg, 3, (C, H, H), pe, po, pn, pon, "cpu",
                               torch.Generator().manual_seed(1))
    sig, ts = traj["sigmas"], traj["timesteps"]
    for t in range(cfg.num_steps):
        x, xn = traj["x"][t], traj["x_next"][t]
        v = cfg_velocity(tr, x, ts[t].expand(3), pe.expand(3, -1, -1), po.expand(3, -1),
                         pn.expand(3, -1, -1), pon.expand(3, -1), cfg.guidance)
        mean, std = sde_mean_std(x, v, sig[t], sig[t + 1], sig[1], cfg.noise_level)
        lp = gaussian_logp(xn, mean, std)
        assert torch.allclose(lp, traj["logp"][t], atol=1e-5), f"logp mismatch at step {t}"
        assert std > 0 and torch.isfinite(lp).all()
    assert abs(sig[0] - 1.0) < 1e-6 and sig[-1] == 0.0, sig
    print("SDE log-prob consistent over", cfg.num_steps, "steps; sigmas", [round(s, 3) for s in sig])

    # --- (2) reward goes up: reward = mean of the final latent
    def reward_fn(latents, prompt):
        return latents.reshape(latents.shape[0], -1).mean(dim=1)

    class NoMetrics:
        metrics = {}

    tmp = tempfile.mkdtemp(prefix="sd35ddpo_")
    try:
        torch.manual_seed(0)
        tr = tiny_transformer(); pipe = FakePipe(tr)
        cfg2 = DDPOConfig(num_steps=4, guidance=2.0, noise_level=0.7, resolution=H * 8, k_per_prompt=8,
                          prompts_per_epoch=2, sample_batch=8, epochs=30, train_batch=8, grad_accum=1,
                          learning_rate=1e-2, clip_range=1e-2, adv_clip=5.0, beta_kl=0.0,
                          gradient_checkpointing=False, seed=0, save_every_epochs=0)
        before = {n: p.detach().clone() for n, p in tr.named_parameters() if p.requires_grad}
        logs = []
        hist = train_ddpo(pipe, tr, ["a", "b", "c"], NoMetrics(), cfg2, tmp, device="cpu",
                          reward_fn=reward_fn, log=lambda s: logs.append(s))
        r = [h["reward_mean"] for h in hist]
        first, last = sum(r[:5]) / 5, sum(r[-5:]) / 5
        n = len(r); xm = (n - 1) / 2; ym = sum(r) / n
        slope = sum((i - xm) * (v - ym) for i, v in enumerate(r)) / sum((i - xm) ** 2 for i in range(n))
        print("\n".join(logs[1:4] + ["..."] + logs[-2:]))
        # platform numerics (CPU torch version) change the magnitude; a broken update gives ~0 change and no trend
        assert last > first + 0.01 and slope > 0, f"reward did not increase: {first:.3f} -> {last:.3f} (slope {slope:+.4f})"
        assert all(h["clip_frac"] <= 1.0 and h["loss"] == h["loss"] for h in hist)
        moved = max((p.detach() - before[n]).abs().max().item() for n, p in tr.named_parameters() if p.requires_grad)
        assert moved > 1e-3, "LoRA parameters did not move"
        # --- (3) save / load
        tr2 = tiny_transformer()
        n = load_lora_into_adapter(tr2, tmp)
        assert n > 0
        print(f"reward {first:+.3f} -> {last:+.3f} (slope {slope:+.4f}/epoch); adapter reloaded ({n} tensors)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    identity_wiring_test()
    print("OK — DDPO (Flow-GRPO SDE + PPO) works on a tiny SD3 transformer")


def identity_wiring_test():
    """(4) identity term: tiny CLIP (no download) -> score/calibrate/term math; DDPO with
    identity_lam + adv_mode=global logs raw_identity and the rollouts CSV column."""
    import tempfile, shutil, csv
    from transformers import CLIPConfig, CLIPModel
    from reward.identity import IdentityReward
    cfg_clip = CLIPConfig(text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, vocab_size=100),
                          vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                             image_size=224, patch_size=32), projection_dim=16)
    idm = IdentityReward.__new__(IdentityReward)
    idm.device, idm.topk, idm.batch_size = "cpu", 2, 8
    idm.model = CLIPModel(cfg_clip).eval()
    idm.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    idm.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    idm.tau = idm.sigma = None
    refs = torch.rand(6, 3, 40, 40)
    idm.ref = idm.embed(refs)
    assert idm.ref.shape == (6, 16) and torch.allclose(idm.ref.norm(dim=1), torch.ones(6), atol=1e-4)
    s_ref = idm.score(refs)                       # references score ~1 against themselves (top-k includes self)
    assert (s_ref > 0.5).all(), s_ref
    calib = torch.rand(10, 3, 40, 40)
    idm.calibrate(calib)
    t = idm.term(torch.tensor([idm.tau + 1.0, idm.tau - idm.sigma]))
    assert t[0].item() == 0.0 and abs(t[1].item() + 1.0) < 1e-3, t

    class RM:
        metrics = {"nima": None}
        def raw_scores(self, imgs): return {"nima": imgs.mean(dim=(1, 2, 3)) * 10}
        def composite_from_raw(self, raw):
            from reward.metrics import composite_from_raw; return composite_from_raw(raw, {"nima": 1.0})
    def decode(z):
        x = torch.nn.functional.interpolate(z[:, :3], size=(40, 40)); return (x.clamp(-2, 2) + 2) / 4
    torch.manual_seed(0)
    tr = tiny_transformer(); pipe = FakePipe(tr); tmp = tempfile.mkdtemp()
    try:
        cfg = DDPOConfig(num_steps=3, guidance=1.0, resolution=H * 8, k_per_prompt=4, prompts_per_epoch=2, sample_batch=4,
                         epochs=3, train_batch=4, learning_rate=1e-3, gradient_checkpointing=False, save_every_epochs=0,
                         save_rollouts=True, rollouts_per_prompt=2, identity_lam=1.0, adv_mode="global", baseline_window=64,
                         shape_lam=5.0, shape_tol=0.05)
        hist = train_ddpo(pipe, tr, ["a", "b"], RM(), cfg, tmp, device="cpu", decode_fn=decode, log=lambda s: None,
                          identity_model=idm)
        assert all("raw_identity" in h and "raw_nima" in h and "raw_shape" in h for h in hist), hist[0].keys()
        rows = list(csv.DictReader(open(os.path.join(tmp, "rollouts", "rollouts.csv"))))
        assert "identity" in rows[0] and "shape" in rows[0] and len(rows) == 3 * 2 * 2, (rows[0].keys(), len(rows))
        # epoch 1: policy == reference at init -> paired rollouts identical -> shape similarity 1
        assert hist[0]["raw_shape"] > 0.99, hist[0]["raw_shape"]
        print(f"identity+shape wiring OK: identity={hist[-1]['raw_identity']:.3f} shape={hist[-1]['raw_shape']:.3f} "
              f"(epoch1 shape={hist[0]['raw_shape']:.3f}), {len(rows)} rollout rows, global baseline")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
