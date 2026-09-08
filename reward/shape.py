"""
Shape / layout term for the reward: how much did the POLICY change the geometry of
the image the REFERENCE model would have produced from the very same noise?

    s = edge_similarity(policy_img, reference_img)  in [0, 1]      (1 = identical edge map)
    shape term = shape_lam * min(0, s - (1 - shape_tol))           <= 0

Edge maps = Sobel gradient magnitude of the grayscale image, lightly blurred;
similarity = Pearson correlation of the two maps (centred, so two unrelated maps
give ~0 instead of the positive bias of a plain cosine), averaged over three
scales (64 / 128 / 256 px) so the outline and hardware layout weigh more than
figured-top texture. Colour and finish barely move it; body outline, headstock,
neck, pickups, bridge, knobs, framing/zoom do. Pure torch, no downloads.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_BLUR = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


def edge_map(images: torch.Tensor, size: int = 256) -> torch.Tensor:
    """(N,3,H,W) in [0,1] -> (N,1,size,size) blurred Sobel magnitude."""
    x = images.float()
    g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    g = F.interpolate(g, size=(size, size), mode="area")
    kx = _SOBEL_X.to(g.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), ky)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    blur = _BLUR.to(g.device).view(1, 1, 3, 3)
    return F.conv2d(F.pad(mag, (1, 1, 1, 1), mode="replicate"), blur)


def _pearson(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    ea = ea.flatten(1); eb = eb.flatten(1)
    ea = ea - ea.mean(dim=1, keepdim=True); eb = eb - eb.mean(dim=1, keepdim=True)
    return F.cosine_similarity(ea, eb, dim=1, eps=1e-8)


@torch.no_grad()
def edge_similarity(a: torch.Tensor, b: torch.Tensor, sizes=(64, 128, 256)) -> torch.Tensor:
    """(N,) multi-scale Pearson correlation of the edge maps of two image batches, clamped to [0, 1]."""
    sims = [_pearson(edge_map(a, s), edge_map(b, s)) for s in sizes]
    return torch.stack(sims, dim=0).mean(dim=0).clamp(0, 1)


def shape_term(sim: torch.Tensor, tol: float) -> torch.Tensor:
    """Penalty <= 0: free within `tol` of a perfect match, linear below."""
    return torch.clamp(sim - (1.0 - tol), max=0.0)
