#!/usr/bin/env python3
"""
CPU smoke test of the flow-matching DPO step on a TINY random SD3 transformer.
No weights are downloaded. Run BEFORE loading SD-3.5 on Colab:

    python tests/smoke_tiny.py

Checks: LoRA target modules attach to SD3Transformer2DModel, adapter on/off
switches the reference, shapes, the loss is finite, gradients reach LoRA
params only, and one optimiser step lowers the loss on a fixed batch.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel  # noqa: E402
from diffusers.training_utils import cast_training_params  # noqa: E402
from peft import LoraConfig  # noqa: E402

from train.dpo import (DPOConfig, SD3_LORA_TARGET_MODULES, dpo_step, dpo_loss,  # noqa: E402
                       load_lora_into_adapter, save_lora)


def main():
    torch.manual_seed(0)
    C, H, heads, hd = 16, 8, 2, 8
    tr = SD3Transformer2DModel(
        sample_size=H, patch_size=2, in_channels=C, out_channels=C, num_layers=2,
        attention_head_dim=hd, num_attention_heads=heads,
        joint_attention_dim=32, caption_projection_dim=heads * hd,
        pooled_projection_dim=24, pos_embed_max_size=16,
    )
    tr.requires_grad_(False)
    tr.add_adapter(LoraConfig(r=4, lora_alpha=8, init_lora_weights="gaussian",
                              target_modules=SD3_LORA_TARGET_MODULES))
    cast_training_params(tr, dtype=torch.float32)
    n_lora = sum(p.numel() for p in tr.parameters() if p.requires_grad)
    assert n_lora > 0, "no LoRA params attached — target modules wrong"
    print(f"LoRA params: {n_lora}")

    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    cfg = DPOConfig(beta=5.0, learning_rate=1e-3, lora_rank=4, lora_alpha=8,
                    gradient_checkpointing=False)
    B, L = 2, 6
    lat_w = torch.randn(B, C, H, H); lat_l = torch.randn(B, C, H, H)
    pe = torch.randn(B, L, 32); po = torch.randn(B, 24)

    # sanity on the objective itself
    l0, a0 = dpo_loss(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]), 5.0)
    assert abs(l0.item() - 0.6931) < 1e-3 and a0.item() == 0.0, "loss at ref == policy must be log 2"
    l1, a1 = dpo_loss(torch.tensor([0.5]), torch.tensor([1.5]), torch.tensor([1.0]), torch.tensor([1.0]), 5.0)
    assert l1 < l0 and a1.item() == 1.0, "policy that prefers chosen must lower the loss"

    # reference == policy at init? (gaussian init on A only -> B is zero -> yes)
    g = torch.Generator().manual_seed(1)
    loss, st = dpo_step(tr, sched, lat_w, lat_l, pe, po, cfg, g)
    assert torch.isfinite(loss), "non-finite loss"
    assert abs(st["pol_w"] - st["ref_w"]) < 1e-5, "at init the policy must equal the reference"
    print("step0:", {k: round(v, 4) for k, v in st.items()})

    loss0 = loss.item()
    loss.backward()
    grads = [(n, p.grad) for n, p in tr.named_parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for _, g in grads), "no gradient on LoRA"
    assert all(p.grad is None for n, p in tr.named_parameters() if not p.requires_grad), "grad leaked to base"

    opt = torch.optim.AdamW([p for p in tr.parameters() if p.requires_grad], lr=cfg.learning_rate)
    for i in range(30):
        opt.zero_grad()
        g = torch.Generator().manual_seed(1)          # same noise/timesteps each time
        loss, st = dpo_step(tr, sched, lat_w, lat_l, pe, po, cfg, g)
        loss.backward(); opt.step()
    print("step30:", {k: round(v, 4) for k, v in st.items()})
    assert st["loss"] < loss0 - 1e-4, "loss did not decrease on a fixed batch"

    # save -> fresh adapter -> load_lora_into_adapter must reproduce the trained weights
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="sd35dpo_lora_")
    try:
        save_lora(tr, tmp)
        trained = {n: p.detach().clone() for n, p in tr.named_parameters() if p.requires_grad}
        tr2 = SD3Transformer2DModel(
            sample_size=H, patch_size=2, in_channels=C, out_channels=C, num_layers=2,
            attention_head_dim=hd, num_attention_heads=heads,
            joint_attention_dim=32, caption_projection_dim=heads * hd,
            pooled_projection_dim=24, pos_embed_max_size=16,
        )
        tr2.load_state_dict({k: v for k, v in tr.state_dict().items() if "lora_" not in k}, strict=False)
        tr2.requires_grad_(False)
        tr2.add_adapter(LoraConfig(r=4, lora_alpha=8, init_lora_weights="gaussian",
                                   target_modules=SD3_LORA_TARGET_MODULES))
        n = load_lora_into_adapter(tr2, tmp)
        assert n > 0
        for name, p in tr2.named_parameters():
            if p.requires_grad:
                assert torch.allclose(p.detach(), trained[name].to(p.dtype), atol=1e-6), f"mismatch after reload: {name}"
        print(f"save/load round trip OK ({n} tensors)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("OK — flow-matching DPO step works on a tiny SD3 transformer")


if __name__ == "__main__":
    main()
