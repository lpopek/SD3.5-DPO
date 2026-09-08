"""
Diffusion-DPO for Stable Diffusion 3.5 (MMDiT, flow matching) with LoRA.

Adaptation of Diffusion-DPO (Wallace et al., 2023; diffusers
examples/research_projects/diffusion_dpo) from UNet/DDPM to the SD-3 family:

  * noise schedule:  FlowMatchEulerDiscreteScheduler
                     x_t = (1 - sigma) * x_0 + sigma * eps
  * model output:    velocity  v = eps - x_0        (same target as
                     diffusers/examples/dreambooth/train_dreambooth_lora_sd3.py)
  * per-sample error: e = mean_over_dims( (v_pred - v_target)^2 )
  * DPO objective (same convention as the diffusers DPO script):
        model_diff = e_policy(chosen) - e_policy(rejected)
        ref_diff   = e_ref(chosen)    - e_ref(rejected)
        loss       = -logsigmoid( beta * (ref_diff - model_diff) )

Reference policy
----------------
The DPO adapter is trained ON TOP of the "plain" personalisation LoRA:
the plain LoRA is fused into the transformer weights, then a fresh LoRA
adapter is added. Disabling that adapter yields the reference model, so
`plain` vs `plain + DPO` isolates the effect of reward fine-tuning.

Everything here is architecture-specific to SD-3/3.5 (MMDiT) but does not
depend on the reward axis: the axis lives entirely in the preference data.
"""
from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class DPOConfig:
    # LoRA (Diffusion-DPO on SD-3.5-M, arXiv:2605.19839 as cited in the handoff)
    lora_rank: int = 32
    lora_alpha: int = 64
    # optimisation
    learning_rate: float = 2.56e-6
    beta: float = 100.0
    max_steps: int = 500
    batch_pairs: int = 1            # preference pairs per micro-batch
    grad_accum: int = 4             # effective batch = batch_pairs * grad_accum
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    lr_warmup_steps: int = 0
    # data
    resolution: int = 1024
    max_sequence_length: int = 256  # T5 tokens (SD-3.5 default 256)
    # timestep sampling (SD-3 default: logit-normal, mean 0, std 1)
    timestep_sampling: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    # runtime
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    seed: int = 0
    log_every: int = 10
    save_every: int = 100

    @classmethod
    def from_dict(cls, d: dict) -> "DPOConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


# SD-3 / SD-3.5 MMDiT attention projections (joint attention: image + text streams)
SD3_LORA_TARGET_MODULES = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]


# ----------------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------------
def build_policy(base_model: str, plain_lora: str | None, cfg: DPOConfig,
                 device: str = "cuda", dtype=torch.bfloat16, init_lora: str | None = None):
    """
    Load SD-3.5, fuse the plain personalisation LoRA into the transformer,
    then attach a fresh trainable LoRA adapter (the DPO policy).
    `init_lora`: folder of a previously saved DPO adapter (same rank/alpha) to
    continue training from instead of a gaussian init.

    Returns (pipe, transformer). `transformer.disable_adapters()` gives the
    reference model (plain), `enable_adapters()` the policy.
    """
    from diffusers import StableDiffusion3Pipeline
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    pipe = StableDiffusion3Pipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.to(device)

    if plain_lora:
        pipe.load_lora_weights(plain_lora)
        pipe.fuse_lora(components=["transformer"])
        pipe.unload_lora_weights()          # weights stay fused, adapter bookkeeping removed
        print(f"[dpo] plain LoRA fused into transformer: {plain_lora}")

    transformer = pipe.transformer
    transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    for te in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        if te is not None:
            te.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        init_lora_weights="gaussian", target_modules=SD3_LORA_TARGET_MODULES,
    )
    transformer.add_adapter(lora_cfg)
    if init_lora:
        load_lora_into_adapter(transformer, init_lora)
    # LoRA params must be fp32: with LR ~1e-6 a bf16 update underflows to zero.
    cast_training_params(transformer, dtype=torch.float32)

    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"[dpo] trainable LoRA params: {n_train/1e6:.2f}M")
    return pipe, transformer


# ----------------------------------------------------------------------------
# conditioning + latents
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_prompts(pipe, prompts: list[str], max_sequence_length: int = 256,
                   batch_size: int = 8) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Encode every unique prompt once with the three SD-3.5 text encoders.
    Returns {prompt: (prompt_embeds[1,L,D] cpu, pooled[1,P] cpu)}.
    """
    cache = {}
    uniq = sorted(set(prompts))
    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i:i + batch_size]
        pe, _, pooled, _ = pipe.encode_prompt(
            prompt=chunk, prompt_2=None, prompt_3=None,
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
            device=pipe.device,
        )
        for j, p in enumerate(chunk):
            cache[p] = (pe[j:j + 1].cpu(), pooled[j:j + 1].cpu())
    return cache


def free_text_encoders(pipe):
    """Drop the text encoders (T5-XXL alone is ~9.5 GB in bf16) once prompts are cached."""
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        if getattr(pipe, name, None) is not None:
            setattr(pipe, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def encode_images(vae, pixels: torch.Tensor) -> torch.Tensor:
    """pixels (N,3,H,W) in [-1,1] -> SD-3 latents (N,16,H/8,W/8), shifted+scaled."""
    pixels = pixels.to(device=vae.device, dtype=vae.dtype)
    lat = vae.encode(pixels).latent_dist.sample()
    lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
    return lat


# ----------------------------------------------------------------------------
# flow matching
# ----------------------------------------------------------------------------
def sample_timesteps(scheduler, n: int, cfg: DPOConfig, device, generator=None):
    """
    Sample training timesteps as in train_dreambooth_lora_sd3.py.
    Returns (timesteps[n], sigmas[n]) — sigmas already include the SD-3.5 shift.
    """
    if cfg.timestep_sampling == "logit_normal":
        u = torch.normal(mean=cfg.logit_mean, std=cfg.logit_std, size=(n,),
                         generator=generator, device="cpu")
        u = torch.sigmoid(u)
    elif cfg.timestep_sampling == "uniform":
        u = torch.rand((n,), generator=generator, device="cpu")
    else:
        raise ValueError(f"unknown timestep_sampling: {cfg.timestep_sampling}")
    n_train = scheduler.config.num_train_timesteps
    idx = (u * n_train).long().clamp(max=n_train - 1)
    # scheduler.timesteps[i] == scheduler.sigmas[i] * num_train_timesteps for the
    # training grid, so a shared index keeps them aligned.
    timesteps = scheduler.timesteps[idx].to(device)
    sigmas = scheduler.sigmas[idx].to(device)
    return timesteps, sigmas


def flow_matching_losses(transformer, latents: torch.Tensor, noise: torch.Tensor,
                         timesteps: torch.Tensor, sigmas: torch.Tensor,
                         prompt_embeds: torch.Tensor, pooled: torch.Tensor,
                         autocast_dtype=torch.bfloat16) -> torch.Tensor:
    """
    Per-sample velocity-prediction error for SD-3.5.

    x_t     = (1 - sigma) * x_0 + sigma * eps
    target  = eps - x_0                       (velocity, as in the SD3 trainer)
    returns mean_{C,H,W} (v_pred - target)^2  -> shape (N,)
    """
    s = sigmas.view(-1, *([1] * (latents.ndim - 1))).to(latents.dtype)
    noisy = (1.0 - s) * latents + s * noise
    with torch.autocast(device_type=latents.device.type, dtype=autocast_dtype,
                        enabled=latents.device.type == "cuda"):
        pred = transformer(
            hidden_states=noisy.to(autocast_dtype if latents.device.type == "cuda" else latents.dtype),
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]
    target = noise - latents
    err = (pred.float() - target.float()) ** 2
    return err.reshape(err.shape[0], -1).mean(dim=1)


def dpo_loss(policy_err_chosen, policy_err_rejected,
             ref_err_chosen, ref_err_rejected, beta: float):
    """
    Diffusion-DPO objective from per-sample errors (lower error = higher
    implicit log-likelihood). Identical to the diffusers DPO script:
        logits = ref_diff - model_diff ; loss = -logsigmoid(beta * logits)
    Returns (loss, implicit_acc) where implicit_acc = fraction of pairs with
    logits > 0 (policy prefers chosen more than the reference does).
    """
    model_diff = policy_err_chosen - policy_err_rejected
    ref_diff = ref_err_chosen - ref_err_rejected
    logits = ref_diff - model_diff
    loss = -F.logsigmoid(beta * logits).mean()
    acc = (logits > 0).float().mean().detach()
    return loss, acc


def dpo_step(transformer, scheduler, latents_w: torch.Tensor, latents_l: torch.Tensor,
             prompt_embeds: torch.Tensor, pooled: torch.Tensor, cfg: DPOConfig,
             generator=None):
    """
    One DPO forward for a batch of B pairs. Chosen and rejected share the same
    noise and timestep (essential: the comparison must be at the same t).
    latents_*: (B,16,h,w); prompt_embeds: (B,L,D); pooled: (B,P) for the B prompts.
    Returns (loss, stats dict).
    """
    B = latents_w.shape[0]
    device = latents_w.device
    latents = torch.cat([latents_w, latents_l], dim=0)              # (2B, ...)
    noise = torch.randn(latents_w.shape, generator=generator, device="cpu",
                        dtype=latents.dtype).to(device).repeat(2, 1, 1, 1)
    t, s = sample_timesteps(scheduler, B, cfg, device, generator)
    t, s = t.repeat(2), s.repeat(2)
    pe = torch.cat([prompt_embeds, prompt_embeds], dim=0)
    po = torch.cat([pooled, pooled], dim=0)

    # policy (adapter ON)
    transformer.enable_adapters()
    err = flow_matching_losses(transformer, latents, noise, t, s, pe, po)
    pol_w, pol_l = err.chunk(2)

    # reference (adapter OFF) — same noise / timesteps
    with torch.no_grad():
        transformer.disable_adapters()
        ref = flow_matching_losses(transformer, latents, noise, t, s, pe, po)
        transformer.enable_adapters()
    ref_w, ref_l = ref.chunk(2)

    loss, acc = dpo_loss(pol_w, pol_l, ref_w, ref_l, cfg.beta)
    stats = dict(
        loss=loss.item(), implicit_acc=acc.item(),
        pol_w=pol_w.mean().item(), pol_l=pol_l.mean().item(),
        ref_w=ref_w.mean().item(), ref_l=ref_l.mean().item(),
        sigma_mean=s[:B].mean().item(),
    )
    return loss, stats


# ----------------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------------
def train_dpo(pipe, transformer, dataset, cfg: DPOConfig, out_dir: str,
              prompt_cache: dict | None = None, device: str = "cuda"):
    """
    Full Diffusion-DPO training on a PreferenceDataset (see train/data.py).

    dataset[i] -> {"prompt": str, "chosen": (3,H,W) in [-1,1], "rejected": (3,H,W)}
    Prompts are encoded once (prompt_cache) and the text encoders are freed.
    Saves the LoRA adapter to out_dir (diffusers format, loadable with
    pipe.load_lora_weights) every cfg.save_every steps and at the end.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if prompt_cache is None:
        print("[dpo] encoding prompts ...")
        prompt_cache = encode_prompts(pipe, [dataset.prompt_at(i) for i in range(len(dataset))],
                                      cfg.max_sequence_length)
        free_text_encoders(pipe)

    params = [p for p in transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.999),
                            weight_decay=cfg.weight_decay, eps=1e-8)

    def lr_lambda(step):
        if cfg.lr_warmup_steps > 0 and step < cfg.lr_warmup_steps:
            return (step + 1) / cfg.lr_warmup_steps
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_pairs, shuffle=True,
                        drop_last=True, num_workers=2, collate_fn=dataset.collate,
                        generator=gen)

    transformer.train()
    step, micro, t0 = 0, 0, time.time()
    running = {}
    history = []
    done = False
    while not done:
        for batch in loader:
            with torch.no_grad():
                lat_w = encode_images(pipe.vae, batch["chosen"])
                lat_l = encode_images(pipe.vae, batch["rejected"])
            pe = torch.cat([prompt_cache[p][0] for p in batch["prompt"]]).to(device)
            po = torch.cat([prompt_cache[p][1] for p in batch["prompt"]]).to(device)

            loss, stats = dpo_step(transformer, scheduler, lat_w, lat_l, pe, po, cfg, gen)
            (loss / cfg.grad_accum).backward()
            micro += 1
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v / cfg.grad_accum

            if micro % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                running["step"] = step
                history.append(dict(running))
                if step % cfg.log_every == 0 or step == 1:
                    el = time.time() - t0
                    win = history[-cfg.log_every:]          # mean over the last log window
                    m = {k: sum(h[k] for h in win) / len(win) for k in win[0] if k != "step"}
                    # policy-vs-reference margin on the window: >0 means the policy
                    # separates chosen from rejected better than the reference
                    margin = (m["ref_w"] - m["ref_l"]) - (m["pol_w"] - m["pol_l"])
                    print(f"[dpo] step {step}/{cfg.max_steps} loss={m['loss']:.4f} "
                          f"acc={m['implicit_acc']:.2f} margin={margin:+.5f} "
                          f"pol(w/l)={m['pol_w']:.4f}/{m['pol_l']:.4f} "
                          f"ref(w/l)={m['ref_w']:.4f}/{m['ref_l']:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} {el/step:.1f}s/step  [mean of {len(win)} steps]")
                running = {}
                if cfg.save_every and step % cfg.save_every == 0:
                    save_lora(transformer, os.path.join(out_dir, f"step_{step:05d}"))
                if step >= cfg.max_steps:
                    done = True
                    break

    save_lora(transformer, out_dir)
    _dump_json(os.path.join(out_dir, "train_log.json"),
               dict(config=asdict(cfg), history=history))
    return history


# ----------------------------------------------------------------------------
# save / load
# ----------------------------------------------------------------------------
def save_lora(transformer, out_dir: str):
    """Save the DPO LoRA in diffusers format (pytorch_lora_weights.safetensors)."""
    from diffusers import StableDiffusion3Pipeline
    from peft.utils import get_peft_model_state_dict
    os.makedirs(out_dir, exist_ok=True)
    sd = get_peft_model_state_dict(transformer)
    StableDiffusion3Pipeline.save_lora_weights(save_directory=out_dir,
                                               transformer_lora_layers=sd)
    print(f"[dpo] LoRA saved: {out_dir}")


def load_policy_pipeline(base_model: str, plain_lora: str | None, dpo_lora: str | None,
                         device: str = "cuda", dtype=torch.bfloat16, dpo_scale: float = 1.0):
    """
    Inference pipeline for plain (dpo_lora=None) or plain+DPO generations.
    Plain LoRA is fused exactly as in training, so 'plain' and 'rl_*' differ
    ONLY by the DPO adapter. `dpo_scale` multiplies the DPO adapter (1.0 = as
    trained; >1 amplifies the learned shift — an inference-time ablation).
    """
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.to(device)
    if plain_lora:
        pipe.load_lora_weights(plain_lora)
        pipe.fuse_lora(components=["transformer"])
        pipe.unload_lora_weights()
    if dpo_lora:
        pipe.load_lora_weights(dpo_lora, adapter_name="dpo")
        if dpo_scale != 1.0:
            pipe.set_adapters(["dpo"], adapter_weights=[float(dpo_scale)])
            print(f"[dpo] adapter scale = {dpo_scale}")
    return pipe


def load_lora_into_adapter(transformer, lora_dir: str):
    """
    Load a DPO adapter saved by save_lora() into the (already attached) peft
    adapter of `transformer` — used to continue training. Keys in the diffusers
    file are 'transformer.<module>.lora_A.weight'; peft wants '<module>.lora_A.weight'.
    """
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file
    path = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
    sd = load_file(path)
    sd = {(k[len("transformer."):] if k.startswith("transformer.") else k): v for k, v in sd.items()}
    res = set_peft_model_state_dict(transformer, sd)
    unexpected = getattr(res, "unexpected_keys", [])
    n_loaded = len(sd) - len(unexpected)
    if n_loaded == 0:
        raise ValueError(f"no LoRA tensors matched the adapter — wrong rank/targets? ({path})")
    print(f"[dpo] init from {path}: {n_loaded} tensors" + (f", {len(unexpected)} unexpected" if unexpected else ""))
    return n_loaded


def _dump_json(path, obj):
    import json
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
