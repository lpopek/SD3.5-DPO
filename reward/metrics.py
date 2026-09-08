"""
Metric-based reward (pyiqa NR-IQA), split into two axes:

  AESTHETIC  (rl_aes):  nima + laion_aes
  TECHNICAL  (rl_tech): musiq + topiq_nr

CLIP-IQA (sharp/blurry prompt pair) is optional and belongs to the technical
axis ONLY; it was dropped from the defaults on 2026-09-03 because it saturates
(0.997 +- 0.003 over the pool), which turns its within-prompt robust-z into noise.
Pass weights={"musiq":1,"topiq_nr":1,"clipiqa":1} to bring it back.

Composite = weighted mean of robust-z (median / MAD) scores. Weights are
interchangeable (a hook for later MOS calibration). All metrics are
"higher is better". pyiqa interface: create_metric(name, device) -> callable
on a tensor (N,3,H,W), RGB in [0,1].

`robust_z` / `composite_from_raw` are module-level so that pair building from
a cached scores.csv (scripts/build_pairs.py) needs neither pyiqa nor a GPU.
"""
from __future__ import annotations

import torch

AESTHETIC_METRICS = {
    "nima": 1.0,        # NIMA (AVA) — predicted aesthetic score distribution
    "laion_aes": 1.0,   # LAION-Aesthetics v2 — linear head on CLIP
}
TECHNICAL_METRICS = {
    "musiq": 1.0,       # multi-scale transformer, technical quality
    "topiq_nr": 1.0,    # top-down semantics -> distortions
    # "clipiqa": 1.0,   # CLIP-IQA "Sharp photo." / "Blurry photo." — optional, saturates here
}
AXES = {"aesthetic": AESTHETIC_METRICS, "technical": TECHNICAL_METRICS}
CLIPIQA_SHARP_PROMPTS = ["Sharp photo.", "Blurry photo."]


def robust_z(x: torch.Tensor) -> torch.Tensor:
    """(x - median) / (1.4826 * MAD); MAD floored so a constant vector gives 0."""
    x = x.float()
    med = x.median()
    mad = (x - med).abs().median()
    mad = mad if mad > 1e-9 else torch.tensor(1e-9)
    return (x - med) / (1.4826 * mad)


def composite_from_raw(raw: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """Weighted mean of per-metric robust-z scores. Standardisation is WITHIN the
    given batch, so call it on all K generations of one prompt at once."""
    total, wsum = None, 0.0
    for name, w in weights.items():
        z = robust_z(raw[name]) * w
        total = z if total is None else total + z
        wsum += w
    return total / max(wsum, 1e-9)


class RewardModel:
    """
    rm = RewardModel(axis="technical", device="cuda")
    r  = rm.score(images)               # (N,) composite reward, images (N,3,H,W) in [0,1]
    raw = rm.raw_scores(images)         # {metric: (N,)}
    """

    def __init__(self, axis: str = "aesthetic", device: str = "cuda",
                 weights: dict | None = None):
        import pyiqa
        assert axis in AXES, "axis: 'aesthetic' | 'technical'"
        self.axis = axis
        self.device = device
        self.weights = dict(weights) if weights else dict(AXES[axis])
        if axis == "aesthetic" and "clipiqa" in self.weights:
            raise ValueError("clipiqa is a technical-axis metric; remove it from the aesthetic axis")

        self.metrics = {}
        for name in self.weights:
            m = pyiqa.create_metric(name, device=device)
            if name == "clipiqa":
                m = self._patch_clipiqa_prompts(m, CLIPIQA_SHARP_PROMPTS)
            self.metrics[name] = m

    @staticmethod
    def _patch_clipiqa_prompts(metric, prompts):
        """
        Replace the default ("Good photo.", "Bad photo.") pair with a sharpness pair.
        pyiqa's CLIPIQA keeps the tokenised pair in `net.prompt_pairs`; if the
        installed version differs, we keep the default pair and say so loudly.
        """
        try:
            import clip  # pyiqa vendors/depends on openai-clip
            net = getattr(metric, "net", None)
            if net is not None and hasattr(net, "prompt_pairs"):
                tok = clip.tokenize(prompts).to(net.prompt_pairs.device)
                net.prompt_pairs = tok
                print(f"[reward] clipiqa prompts set to {prompts}")
                return metric
        except Exception as e:  # pragma: no cover
            print(f"[reward] WARNING: could not patch clipiqa prompts ({e})")
        print("[reward] WARNING: clipiqa uses default Good/Bad prompts (general quality)")
        return metric

    @torch.no_grad()
    def raw_scores(self, images: torch.Tensor, batch_size: int = 4) -> dict[str, torch.Tensor]:
        images = images.to(self.device)
        out = {}
        for name, metric in self.metrics.items():
            vals = []
            for i in range(0, images.shape[0], batch_size):
                vals.append(metric(images[i:i + batch_size]).flatten().float().cpu())
            out[name] = torch.cat(vals)
        return out

    _robust_z = staticmethod(robust_z)   # backwards compatibility

    def composite_from_raw(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        return composite_from_raw(raw, self.weights)

    @torch.no_grad()
    def score(self, images: torch.Tensor) -> torch.Tensor:
        """
        Composite reward (N,). NOTE: robust-z is computed WITHIN the given batch —
        call it on all K generations of one prompt at once (K >= 6 recommended).
        """
        return self.composite_from_raw(self.raw_scores(images))
