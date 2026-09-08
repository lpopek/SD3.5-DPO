"""
Preference pairs for Diffusion-DPO from per-image rewards.

For one prompt we have K generations and their composite rewards (one axis).
A pair is (chosen, rejected) with reward[chosen] > reward[rejected].

Strategies
----------
top_bottom        ladder: best vs worst, 2nd best vs 2nd worst, ... (at most K//2 rungs)
                  -> the pairing described in the paper (K=8, 2 rungs = 356 pairs)
ranked_gap        every pair (a, b) whose RANK gap is >= min_rank_gap and whose
                  reward margin is >= margin, strongest margins first, up to max_pairs
                  -> denser supervision from the same pool (K=16, gap 4 -> up to 16/prompt)
all_above_margin  every pair with margin >= margin in index order, up to max_pairs
"""
from __future__ import annotations

from itertools import combinations

import torch

STRATEGIES = ("top_bottom", "ranked_gap", "all_above_margin")


def build_preference_pairs(
    images,                    # (K,3,H,W) tensor OR the int K — only K is used
    rewards: torch.Tensor,     # (K,) composite reward per generation
    prompt: str,
    strategy: str = "top_bottom",
    margin: float = 0.0,       # minimum reward difference for a valid pair
    max_pairs: int = 1,        # pairs per prompt
    min_rank_gap: int = 1,     # ranked_gap only: minimum distance in the ranking
) -> list[dict]:
    """
    Returns [{prompt, chosen_idx, rejected_idx, chosen_reward, rejected_reward}];
    indices refer to the rows of `images` / `rewards`.
    """
    K = images if isinstance(images, int) else images.shape[0]
    rewards = torch.as_tensor(rewards).float()
    assert rewards.shape[0] == K, "rewards must have length K"
    order = torch.argsort(rewards, descending=True).tolist()   # best first

    pairs = []
    if strategy == "top_bottom":
        for i in range(min(max_pairs, K // 2)):
            hi, lo = order[i], order[K - 1 - i]
            if (rewards[hi] - rewards[lo]).item() >= margin:
                pairs.append(_pair(prompt, hi, lo, rewards))
    elif strategy == "ranked_gap":
        rank = {idx: r for r, idx in enumerate(order)}
        cands = []
        for a, b in combinations(range(K), 2):
            hi, lo = (a, b) if rewards[a] >= rewards[b] else (b, a)
            gap = abs(rank[a] - rank[b])
            m = (rewards[hi] - rewards[lo]).item()
            if gap >= min_rank_gap and m >= margin:
                cands.append((m, gap, hi, lo))
        cands.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
        for m, gap, hi, lo in cands[:max_pairs]:
            pairs.append(_pair(prompt, hi, lo, rewards))
    elif strategy == "all_above_margin":
        for a, b in combinations(range(K), 2):
            hi, lo = (a, b) if rewards[a] >= rewards[b] else (b, a)
            if (rewards[hi] - rewards[lo]).item() >= margin:
                pairs.append(_pair(prompt, hi, lo, rewards))
                if len(pairs) >= max_pairs:
                    break
    else:
        raise ValueError(f"unknown strategy: {strategy} (choose from {STRATEGIES})")
    return pairs


def _pair(prompt, hi, lo, rewards):
    return {
        "prompt": prompt,
        "chosen_idx": int(hi),
        "rejected_idx": int(lo),
        "chosen_reward": float(rewards[hi]),
        "rejected_reward": float(rewards[lo]),
    }
