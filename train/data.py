"""
Preference dataset for Diffusion-DPO.

Reads `preferences.jsonl` written by data/generate.py — one JSON per line:
    {"prompt": str, "chosen": path, "rejected": path,
     "chosen_reward": float, "rejected_reward": float}

Images are resized/center-cropped to `resolution` and scaled to [-1, 1]
(the input range of the SD-3 VAE).
"""
from __future__ import annotations

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_image(path: str, resolution: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = resolution / min(w, h)
    im = im.resize((max(resolution, round(w * s)), max(resolution, round(h * s))),
                   Image.BICUBIC)
    w, h = im.size
    left, top = (w - resolution) // 2, (h - resolution) // 2
    im = im.crop((left, top, left + resolution, top + resolution))
    import numpy as np
    t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1.0
    return t


class PreferenceDataset(Dataset):
    def __init__(self, jsonl_path: str, resolution: int = 1024,
                 min_margin: float = 0.0, root: str | None = None):
        self.resolution = resolution
        self.root = root or os.path.dirname(os.path.abspath(jsonl_path))
        self.items = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["chosen_reward"] - rec["rejected_reward"] < min_margin:
                    continue
                self.items.append(rec)
        if not self.items:
            raise ValueError(f"no preference pairs in {jsonl_path}")

    def _path(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def __len__(self):
        return len(self.items)

    def prompt_at(self, i: int) -> str:
        return self.items[i]["prompt"]

    def __getitem__(self, i: int):
        rec = self.items[i]
        return {
            "prompt": rec["prompt"],
            "chosen": _load_image(self._path(rec["chosen"]), self.resolution),
            "rejected": _load_image(self._path(rec["rejected"]), self.resolution),
        }

    @staticmethod
    def collate(batch):
        return {
            "prompt": [b["prompt"] for b in batch],
            "chosen": torch.stack([b["chosen"] for b in batch]),
            "rejected": torch.stack([b["rejected"] for b in batch]),
        }

    def summary(self) -> str:
        margins = [r["chosen_reward"] - r["rejected_reward"] for r in self.items]
        n_prompts = len({r["prompt"] for r in self.items})
        return (f"{len(self.items)} pairs over {n_prompts} prompts; "
                f"reward margin mean={sum(margins)/len(margins):.3f} "
                f"min={min(margins):.3f} max={max(margins):.3f}")
