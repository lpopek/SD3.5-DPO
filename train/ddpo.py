"""
DDPO-style online RL for Stable Diffusion 3.5 (rectified flow, MMDiT) with LoRA.

DDPO (Black et al., 2023) treats the denoising trajectory as an MDP and applies
PPO to per-step log-probabilities. That needs a STOCHASTIC sampler. The SD-3
Euler sampler is a deterministic ODE, so we use the SDE of Flow-GRPO
(Liu et al., 2025), which has the same marginals as the ODE and Gaussian steps:

    x_{t+dt} ~ N( x_t + [ v + (s_t^2 / 2t) (x_t + (1 - t) v) ] dt ,  s_t^2 |dt| )
    s_t = noise_level * sqrt(t / (1 - t)),   v = v_theta(x_t, t, c) with CFG,
    dt = t_next - t < 0   (sigmas of FlowMatchEulerDiscreteScheduler, shift 3)

Per sample: T SDE steps -> latent -> VAE -> image -> reward r (same RewardModel
as the DPO pairs: composite of robust-z within the K samples of one prompt).
Advantage A = (r - mean_K) / std_K (group-relative, as DDPO's per-prompt
normalisation) or, with adv_mode="global", (r - running mean) / running std over
the last `baseline_window` samples — needed when the reward has an ABSOLUTE part
(identity term, reward/identity.py): a drift shared by the whole group cancels
in the group-relative advantage but not against a running baseline.
Update (PPO-clip, one ratio per (sample, step)):

    ratio = exp( logp_theta(x_{t+dt} | x_t) - logp_old )       (mean over latent dims)
    L = max( -A ratio, -A clip(ratio, 1-eps, 1+eps) )  [+ beta_kl * KL(theta || ref)]

Policy = LoRA adapter on top of the fused personalisation LoRA (train/dpo.build_policy),
so `plain` and `rl_*` differ only by the adapter and the saved adapter loads with
eval/generate_triplets.py exactly like a DPO one. Training samples with T=10 SDE
steps at `resolution`; inference uses the normal ODE sampler (28 steps, 1024^2).
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import torch

from train.dpo import DPOConfig, encode_prompts, free_text_encoders, save_lora


@dataclass
class DDPOConfig:
    # sampling (training rollouts)
    num_steps: int = 10
    guidance: float = 4.5
    noise_level: float = 0.7
    resolution: int = 768
    k_per_prompt: int = 8
    prompts_per_epoch: int = 8
    sample_batch: int = 4
    # optimisation
    epochs: int = 30
    inner_epochs: int = 1
    train_batch: int = 4
    grad_accum: int = 4
    learning_rate: float = 1e-4
    clip_range: float = 1e-4
    adv_clip: float = 5.0
    beta_kl: float = 0.0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    # LoRA (passed to build_policy)
    lora_rank: int = 32
    lora_alpha: int = 64
    gradient_checkpointing: bool = True
    # runtime
    max_sequence_length: int = 256
    seed: int = 0
    save_every_epochs: int = 5
    start_epoch: int = 0
    # keep the training rollouts (images + rewards) as a by-product dataset:
    #   {out_dir}/rollouts/epoch_NNN/p{prompt_idx:03d}_k{k}.png + {out_dir}/rollouts/rollouts.csv
    save_rollouts: bool = False
    rollouts_per_prompt: int = 8          # at most this many of the K samples per prompt
    # identity term: r = z_group(quality) + identity_lam * min(0, (id - tau) / sigma)
    identity_lam: float = 0.0             # 0 = off; 1.0 = one sigma of identity loss costs one sigma of quality
    # shape term (reward/shape.py): paired reference rollout with the SAME noise, Sobel-edge similarity
    #   r += shape_lam * min(0, edge_sim - (1 - shape_tol))   — geometry/layout/framing must not move
    shape_lam: float = 0.0                # 0 = off (no paired reference rollout); 5.0 = 0.1 edge deviation costs 0.5 sigma
    shape_tol: float = 0.05               # free deviation before the penalty starts
    adv_mode: str = "group"               # "group" | "global". Keep "group": with an absolute penalty term a running
                                          # baseline turns every advantage negative once the penalty kicks in and PPO
                                          # then pushes the policy in a random direction (observed: shape 1.0 -> 0.3)
    baseline_window: int = 512            # samples kept for the running baseline

    @classmethod
    def from_dict(cls, d: dict) -> "DDPOConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})

    def dpo_view(self) -> DPOConfig:
        return DPOConfig(lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                         gradient_checkpointing=self.gradient_checkpointing)


# ----------------------------------------------------------------------------
# SDE step (Flow-GRPO) and its Gaussian log-probability
# ----------------------------------------------------------------------------
def sde_mean_std(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float,
                 sigma_max: float, noise_level: float):
    """Mean and (scalar) std of x_{t+dt} given x_t and the velocity v."""
    s = min(sigma, sigma_max)                       # avoid 1 - sigma = 0 at t = 1
    std_t = noise_level * math.sqrt(s / (1.0 - s))
    dt = sigma_next - sigma                         # < 0
    drift = v + (std_t ** 2 / (2.0 * sigma)) * (x + (1.0 - sigma) * v)
    mean = x + drift * dt
    std = std_t * math.sqrt(-dt)
    return mean, std


def gaussian_logp(x: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    """log N(x; mean, std^2 I), MEAN over non-batch dims -> (B,)  (DDPO / Flow-GRPO convention)."""
    lp = -((x - mean) ** 2) / (2.0 * std ** 2) - math.log(std) - 0.5 * math.log(2.0 * math.pi)
    return lp.reshape(lp.shape[0], -1).mean(dim=1)


def cfg_velocity(transformer, x: torch.Tensor, timestep: torch.Tensor, pe, po, pe_neg, po_neg,
                 guidance: float, autocast_dtype=torch.bfloat16) -> torch.Tensor:
    """Classifier-free-guided velocity, cond and uncond batched together."""
    on_cuda = x.device.type == "cuda"
    with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=on_cuda):
        if guidance and guidance != 1.0:
            out = transformer(
                hidden_states=torch.cat([x, x]).to(autocast_dtype if on_cuda else x.dtype),
                timestep=torch.cat([timestep, timestep]),
                encoder_hidden_states=torch.cat([pe_neg, pe]),
                pooled_projections=torch.cat([po_neg, po]),
                return_dict=False)[0].float()
            v_u, v_c = out.chunk(2)
            return v_u + guidance * (v_c - v_u)
        return transformer(hidden_states=x.to(autocast_dtype if on_cuda else x.dtype), timestep=timestep,
                           encoder_hidden_states=pe, pooled_projections=po, return_dict=False)[0].float()


def make_sigmas(scheduler, num_steps: int, device):
    """(sigmas[T+1] incl. final 0, timesteps[T]) of the training grid."""
    scheduler.set_timesteps(num_steps, device=device)
    return scheduler.sigmas.float().tolist(), scheduler.timesteps.float()


@torch.no_grad()
def sample_trajectories(transformer, scheduler, cfg: DDPOConfig, n: int, latent_shape,
                        pe, po, pe_neg, po_neg, device, generator=None, paired_reference: bool = False):
    """
    n SDE rollouts for ONE prompt conditioning (pe/po are (1,...) and get expanded).
    Returns dict with final latents and per-step tensors stored on CPU:
        x[t] (n,...), x_next[t] (n,...), logp[t] (n,), sigmas, timesteps
    With paired_reference=True the REFERENCE model (adapter disabled) is rolled out
    with the identical initial + per-step noises -> `ref_latents` (shape term).
    """
    sigmas, timesteps = make_sigmas(scheduler, cfg.num_steps, device)
    sigma_max = sigmas[1]
    x0 = torch.randn((n, *latent_shape), generator=generator, device="cpu")
    noises = [torch.randn((n, *latent_shape), generator=generator, device="cpu") for _ in range(cfg.num_steps)]
    pe_b, po_b = pe.to(device).expand(n, -1, -1), po.to(device).expand(n, -1)
    pn_b, pon_b = pe_neg.to(device).expand(n, -1, -1), po_neg.to(device).expand(n, -1)

    def rollout(record: bool):
        x = x0.to(device)
        xs, xns, lps = [], [], []
        for i in range(cfg.num_steps):
            t = timesteps[i].expand(n).to(device)
            v = cfg_velocity(transformer, x, t, pe_b, po_b, pn_b, pon_b, cfg.guidance)
            mean, std = sde_mean_std(x, v, sigmas[i], sigmas[i + 1], sigma_max, cfg.noise_level)
            x_next = mean + std * noises[i].to(device)
            if record:
                lps.append(gaussian_logp(x_next, mean, std).cpu())
                xs.append(x.cpu()); xns.append(x_next.cpu())
            x = x_next
        return x, xs, xns, lps

    x, xs, xns, lps = rollout(record=True)
    out = dict(latents=x, x=xs, x_next=xns, logp=lps, sigmas=sigmas, timesteps=timesteps.cpu())
    if paired_reference:
        transformer.disable_adapters()
        try:
            out["ref_latents"] = rollout(record=False)[0]
        finally:
            transformer.enable_adapters()
    return out


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, batch: int = 2) -> torch.Tensor:
    """SD-3 latents -> images (N,3,H,W) in [0,1]."""
    outs = []
    for i in range(0, latents.shape[0], batch):
        z = latents[i:i + batch].to(vae.device, vae.dtype)
        z = z / vae.config.scaling_factor + vae.config.shift_factor
        img = vae.decode(z, return_dict=False)[0].float()
        outs.append(((img.clamp(-1, 1) + 1) / 2).cpu())
    return torch.cat(outs)


# ----------------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------------
def ppo_loss(logp_new, logp_old, adv, clip_range: float):
    ratio = torch.exp(logp_new - logp_old)
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = torch.max(unclipped, clipped).mean()
    clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean()
    return loss, ratio.detach(), clip_frac.detach()


def train_ddpo(pipe, transformer, prompts: list[str], reward_model, cfg: DDPOConfig, out_dir: str,
               device: str = "cuda", decode_fn=None, reward_fn=None, log=print, identity_model=None):
    """
    Online RL loop. `reward_model` exposes raw_scores(images)->{m:(N,)} and
    composite_from_raw(raw)->(N,) (reward/metrics.RewardModel); `decode_fn` /
    `reward_fn` override VAE decoding and scoring (used by the tiny CPU test).
    `identity_model` (reward/identity.IdentityReward, calibrated) adds
    cfg.identity_lam * term(identity) to the reward and logs `raw_identity`.
    Saves the LoRA adapter (diffusers format) to out_dir and train_log.json.
    """
    from collections import deque
    from diffusers import FlowMatchEulerDiscreteScheduler
    os.makedirs(out_dir, exist_ok=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    rng = random.Random(cfg.seed)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    log("[ddpo] encoding prompts ...")
    cache = encode_prompts(pipe, prompts + [""], cfg.max_sequence_length)
    free_text_encoders(pipe)
    pe_neg, po_neg = cache[""]

    params = [p for p in transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.999),
                            weight_decay=cfg.weight_decay, eps=1e-8)
    lat_ch = pipe.transformer.config.in_channels
    lat_hw = cfg.resolution // 8
    decode = decode_fn or (lambda z: decode_latents(pipe.vae, z))
    metric_names = list(getattr(reward_model, "metrics", {}).keys())
    use_identity = identity_model is not None and cfg.identity_lam > 0
    use_shape = cfg.shape_lam > 0
    if use_shape:
        from reward.shape import edge_similarity, shape_term
    log_names = metric_names + (["identity"] if use_identity else []) + (["shape"] if use_shape else [])
    baseline = deque(maxlen=cfg.baseline_window)

    history = []
    t0 = time.time()
    prompt_index = {p: i for i, p in enumerate(prompts)}
    roll_csv = os.path.join(out_dir, "rollouts", "rollouts.csv")
    if cfg.save_rollouts:
        os.makedirs(os.path.dirname(roll_csv), exist_ok=True)
        if not os.path.exists(roll_csv):
            with open(roll_csv, "w", encoding="utf-8") as f:
                f.write(",".join(["epoch", "prompt_idx", "k", "path", "reward", "advantage", *log_names, "prompt"]) + "\n")
    for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.epochs):
        # ---------------- rollouts
        transformer.eval()
        ep_prompts = rng.sample(prompts, min(cfg.prompts_per_epoch, len(prompts)))
        samples = []          # per prompt: dict(traj, adv, reward, raw)
        rewards_all, raw_all = [], {m: [] for m in log_names}
        for p in ep_prompts:
            pe, po = cache[p]
            trajs = []
            for b0 in range(0, cfg.k_per_prompt, cfg.sample_batch):
                nb = min(cfg.sample_batch, cfg.k_per_prompt - b0)
                trajs.append(sample_trajectories(transformer, scheduler, cfg, nb, (lat_ch, lat_hw, lat_hw),
                                                 pe, po, pe_neg, po_neg, device, gen,
                                                 paired_reference=use_shape and reward_fn is None))
            traj = dict(x=[torch.cat([t["x"][i] for t in trajs]) for i in range(cfg.num_steps)],
                        x_next=[torch.cat([t["x_next"][i] for t in trajs]) for i in range(cfg.num_steps)],
                        logp=[torch.cat([t["logp"][i] for t in trajs]) for i in range(cfg.num_steps)],
                        sigmas=trajs[0]["sigmas"], timesteps=trajs[0]["timesteps"],
                        latents=torch.cat([t["latents"] for t in trajs]))
            if use_shape and reward_fn is None:
                traj["ref_latents"] = torch.cat([t["ref_latents"] for t in trajs])
            if reward_fn is not None:
                r = reward_fn(traj["latents"], p).float().cpu()
                raw = {}
            else:
                imgs = decode(traj["latents"])
                raw = reward_model.raw_scores(imgs)
                r = reward_model.composite_from_raw(raw).float().cpu()
                if use_identity:
                    ident = identity_model.score(imgs).float().cpu()
                    raw = dict(raw, identity=ident)
                    r = r + cfg.identity_lam * identity_model.term(ident)
                if use_shape:
                    ref_imgs = decode(traj["ref_latents"])
                    sim = edge_similarity(imgs, ref_imgs).float().cpu()
                    raw = dict(raw, shape=sim)
                    r = r + cfg.shape_lam * shape_term(sim, cfg.shape_tol)
                    del ref_imgs
            if cfg.adv_mode == "global" and len(baseline) >= 16:
                b = torch.tensor(list(baseline) + r.tolist())
                adv = ((r - b.mean()) / (b.std() + 1e-4)).clamp(-cfg.adv_clip, cfg.adv_clip)
            else:
                adv = ((r - r.mean()) / (r.std() + 1e-4)).clamp(-cfg.adv_clip, cfg.adv_clip)
            baseline.extend(r.tolist())
            if cfg.save_rollouts and reward_fn is None:
                _save_rollouts(out_dir, roll_csv, epoch + 1, prompt_index[p], p, imgs, r, adv, raw,
                               log_names, cfg.rollouts_per_prompt)
            samples.append(dict(prompt=p, traj=traj, adv=adv, reward=r))
            rewards_all.append(r)
            for m in log_names:
                raw_all[m].append(raw[m].float().cpu())
            del traj["latents"]
            traj.pop("ref_latents", None)

        # ---------------- PPO update over (sample, step) pairs
        transformer.train()
        items = [(si, k, t) for si, s in enumerate(samples) for k in range(cfg.k_per_prompt)
                 for t in range(cfg.num_steps)]
        stats = dict(loss=0.0, kl=0.0, clip_frac=0.0, ratio_dev=0.0)
        n_upd = 0
        sigma_max = samples[0]["traj"]["sigmas"][1]
        for _ in range(cfg.inner_epochs):
            rng.shuffle(items)
            opt.zero_grad(set_to_none=True)
            micro = 0
            for b0 in range(0, len(items), cfg.train_batch):
                batch = items[b0:b0 + cfg.train_batch]
                # group by step so sigma is scalar per micro-batch: split the batch by t
                by_t = {}
                for si, k, t in batch:
                    by_t.setdefault(t, []).append((si, k))
                loss_total = 0.0
                for t, idx in by_t.items():
                    s0 = samples[idx[0][0]]["traj"]
                    x = torch.stack([samples[si]["traj"]["x"][t][k] for si, k in idx]).to(device)
                    xn = torch.stack([samples[si]["traj"]["x_next"][t][k] for si, k in idx]).to(device)
                    lp_old = torch.stack([samples[si]["traj"]["logp"][t][k] for si, k in idx]).to(device)
                    adv = torch.stack([samples[si]["adv"][k] for si, k in idx]).to(device)
                    pe = torch.cat([cache[samples[si]["prompt"]][0] for si, _ in idx]).to(device)
                    po = torch.cat([cache[samples[si]["prompt"]][1] for si, _ in idx]).to(device)
                    pn = pe_neg.to(device).expand(len(idx), -1, -1); pon = po_neg.to(device).expand(len(idx), -1)
                    ts = s0["timesteps"][t].expand(len(idx)).to(device)
                    v = cfg_velocity(transformer, x, ts, pe, po, pn, pon, cfg.guidance)
                    mean, std = sde_mean_std(x, v, s0["sigmas"][t], s0["sigmas"][t + 1], sigma_max, cfg.noise_level)
                    lp = gaussian_logp(xn, mean, std)
                    loss, ratio, cf = ppo_loss(lp, lp_old, adv, cfg.clip_range)
                    if cfg.beta_kl > 0:
                        with torch.no_grad():
                            transformer.disable_adapters()
                            v_ref = cfg_velocity(transformer, x, ts, pe, po, pn, pon, cfg.guidance)
                            transformer.enable_adapters()
                            mean_ref, _ = sde_mean_std(x, v_ref, s0["sigmas"][t], s0["sigmas"][t + 1], sigma_max, cfg.noise_level)
                        kl = ((mean - mean_ref) ** 2).reshape(mean.shape[0], -1).mean(dim=1) / (2.0 * std ** 2)
                        loss = loss + cfg.beta_kl * kl.mean()
                        stats["kl"] += kl.mean().item() * len(idx) / len(batch)
                    (loss * len(idx) / len(batch) / cfg.grad_accum).backward()
                    stats["loss"] += loss.item() * len(idx) / len(batch)
                    stats["clip_frac"] += cf.item() * len(idx) / len(batch)
                    stats["ratio_dev"] += (ratio - 1).abs().mean().item() * len(idx) / len(batch)
                micro += 1
                n_upd += 1
                if micro % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    opt.step(); opt.zero_grad(set_to_none=True)
            if micro % cfg.grad_accum != 0:          # flush the remainder
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True)

        r_cat = torch.cat(rewards_all)
        rec = dict(epoch=epoch + 1, reward_mean=r_cat.mean().item(), reward_std=r_cat.std().item(),
                   **{f"raw_{m}": torch.cat(v).mean().item() for m, v in raw_all.items() if v},
                   **{k: v / max(n_upd, 1) for k, v in stats.items()},
                   n_samples=int(r_cat.numel()), elapsed_min=(time.time() - t0) / 60)
        history.append(rec)
        log(f"[ddpo] epoch {epoch + 1}: reward {rec['reward_mean']:+.3f}±{rec['reward_std']:.3f} "
            + " ".join(f"{m}={rec['raw_' + m]:.3f}" for m in log_names)
            + f" loss={rec['loss']:.4f} clip={rec['clip_frac']:.2f} |ratio-1|={rec['ratio_dev']:.1e}"
            + (f" kl={rec['kl']:.2e}" if cfg.beta_kl > 0 else "") + f" ({rec['elapsed_min']:.1f} min)")
        if cfg.save_every_epochs and (epoch + 1) % cfg.save_every_epochs == 0:
            save_lora(transformer, os.path.join(out_dir, f"epoch_{epoch + 1:03d}"))
            _dump(out_dir, cfg, history)
        del samples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_lora(transformer, out_dir)
    _dump(out_dir, cfg, history)
    return history


def _save_rollouts(out_dir, roll_csv, epoch, pi, prompt, imgs, r, adv, raw, metric_names, n_max):
    """Write up to n_max rollout images of one prompt + one CSV row each (reward, advantage, raw metrics)."""
    import numpy as np
    from PIL import Image
    d = os.path.join(out_dir, "rollouts", f"epoch_{epoch:03d}")
    os.makedirs(d, exist_ok=True)
    with open(roll_csv, "a", encoding="utf-8") as f:
        for k in range(min(n_max, imgs.shape[0])):
            rel = os.path.join(f"epoch_{epoch:03d}", f"p{pi:03d}_k{k}.png")
            arr = (imgs[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(out_dir, "rollouts", rel))
            vals = [str(epoch), str(pi), str(k), rel, f"{r[k].item():.5f}", f"{adv[k].item():.5f}",
                    *[f"{raw[m][k].item():.5f}" for m in metric_names], '"' + prompt.replace('"', "'") + '"']
            f.write(",".join(vals) + "\n")


def _dump(out_dir, cfg, history):
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(dict(config=asdict(cfg), history=history), f, indent=1)
