"""
Product-identity term for the reward: CLIP similarity of a generated image to the
reference photographs of the product (Dataset/gitarki/1024/{whole,body}).

    idm = IdentityReward(reference_dirs=[...], device="cuda")     # ViT-L/14 (OpenAI weights)
    s   = idm.score(images)            # (N,) mean of the top-k cosine similarities to the references
    idm.calibrate(images_of_the_reference_model)   # tau = mean, sigma = std of the UNTUNED model
    t   = idm.term(s)                  # min(0, (s - tau) / sigma)  <= 0  : penalty only below the plain level

Reward used by DDPO:  r = z_group(quality metrics) + lam * term(identity)
The quality part stays relative within the K samples of a prompt (finish-level
predictor bias cancels, as in the DPO pairs); the identity part is ABSOLUTE and
anchored to the reference model's own identity level, so a drift shared by the
whole group is still penalised (with adv_mode=global, see train/ddpo.py).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def list_images(dirs) -> list[str]:
    out = []
    for d in dirs:
        out += sorted(p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                      if p.lower().endswith(IMG_EXT))
    return out


def load_images(paths, size: int | None = None) -> torch.Tensor:
    from PIL import Image
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if size:
            im = im.resize((size, size), Image.BICUBIC)
        ims.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(ims)


class IdentityReward:
    def __init__(self, reference_dirs, device="cuda", model="openai/clip-vit-large-patch14",
                 topk: int = 5, cache_path: str | None = None, batch_size: int = 16):
        from transformers import CLIPModel
        self.device = device
        self.topk = topk
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model, torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32)
        self.model.to(device).eval()
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.tau, self.sigma = None, None
        self.ref = self._reference_embeddings(reference_dirs, cache_path)
        print(f"[identity] {self.ref.shape[0]} reference embeddings ({model}, top-{topk})")

    # ---------------------------------------------------------------- embeddings
    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) in [0,1] -> L2-normalised CLIP image features (N,D) on CPU (float32)."""
        outs = []
        for i in range(0, images.shape[0], self.batch_size):
            x = images[i:i + self.batch_size].to(self.device, torch.float32)
            if x.shape[-1] != x.shape[-2]:                      # centre crop to square
                s = min(x.shape[-2:]); t, l = (x.shape[-2] - s) // 2, (x.shape[-1] - s) // 2
                x = x[..., t:t + s, l:l + s]
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
            x = ((x - self.mean) / self.std).to(self.model.dtype)
            vo = self.model.vision_model(pixel_values=x)          # version-independent (get_image_features
            pooled = vo.pooler_output if hasattr(vo, "pooler_output") else vo[1]   # returns a tensor or an output)
            f = self.model.visual_projection(pooled).float()
            outs.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(outs)

    def _reference_embeddings(self, dirs, cache_path):
        paths = list_images(dirs)
        if not paths:
            raise FileNotFoundError(f"no reference images in {dirs}")
        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            if list(z["paths"]) == paths:
                return torch.from_numpy(z["emb"])
        emb = self.embed(load_images(paths, size=512))
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, paths=np.array(paths), emb=emb.numpy())
        return emb

    # ---------------------------------------------------------------- scores
    @torch.no_grad()
    def score(self, images: torch.Tensor) -> torch.Tensor:
        """(N,) identity score: mean of the top-k cosine similarities to the references."""
        sims = self.embed(images) @ self.ref.T                    # (N,R)
        k = min(self.topk, sims.shape[1])
        return sims.topk(k, dim=1).values.mean(dim=1)

    def calibrate(self, images: torch.Tensor | None = None, scores: torch.Tensor | None = None, path: str | None = None):
        """tau/sigma = mean/std of the identity score of the UNTUNED (reference) model's samples."""
        if scores is None:
            scores = self.score(images)
        self.tau = float(scores.mean()); self.sigma = max(float(scores.std()), 1e-3)
        if path:
            json.dump(dict(tau=self.tau, sigma=self.sigma, n=int(scores.numel()), topk=self.topk),
                      open(path, "w"), indent=1)
        print(f"[identity] calibrated on {scores.numel()} images: tau={self.tau:.4f} sigma={self.sigma:.4f}")
        return self.tau, self.sigma

    def load_calibration(self, path: str):
        d = json.load(open(path)); self.tau, self.sigma = float(d["tau"]), float(d["sigma"])
        return self.tau, self.sigma

    def term(self, scores: torch.Tensor) -> torch.Tensor:
        """Penalty <= 0 in units of the reference spread; zero at or above the plain level."""
        assert self.tau is not None, "call calibrate() or load_calibration() first"
        return torch.clamp((scores - self.tau) / self.sigma, max=0.0)
