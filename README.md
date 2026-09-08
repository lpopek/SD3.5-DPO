# sd35-dpo — Diffusion-DPO on Stable Diffusion 3.5 with metric rewards

Reward fine-tuning of the personalised SD-3.5 LoRA ("plain") with Diffusion-DPO,
where the preference signal comes from no-reference IQA metrics on one of two axes:

| axis | folder / method | reward |
|---|---|---|
| aesthetic | `rl_aes` | NIMA + LAION-Aesthetics v2 |
| technical | `rl_tech` | MUSIQ + TOPIQ-NR (CLIP-IQA sharp/blurry dropped 2026-09-03: saturates at 0.997 ± 0.003, its robust-z was noise; re-add via `metrics:`) |

CLIP-IQA, if re-enabled, belongs **only** to the technical axis. The survey compares, for the same
prompt and seed, `plain` / `rl_aes` / `rl_tech`.

## Layout

```
reward/   metrics.py  RewardModel (pyiqa) + robust_z / composite_from_raw (no pyiqa needed)
          identity.py CLIP similarity to the reference photos -> identity term of the reward
          shape.py    edge-map correlation policy vs paired reference rollout -> shape term
          pairs.py    pair strategies: top_bottom (paper), ranked_gap, all_above_margin
          scoring.py  score a shared pool on BOTH axes, cached in pool/scores.csv
data/     generate.py build_pool (shared pool, image-level resume, legacy import)
                      build_dataset (legacy single-axis path)
train/    dpo.py      flow-matching Diffusion-DPO for MMDiT (policy = LoRA on top of plain)
          ddpo.py     DDPO / Flow-GRPO: SDE rollouts + PPO on per-step log-probs (same LoRA policy)
          data.py     PreferenceDataset (jsonl -> tensors)
eval/     generate_triplets.py  survey images per method (seed_XXXX_aes/_tech next to plain)
          reward_gain.py        plain vs policy on held-out prompts/seeds (the metric that matters)
scripts/  build_pool.py         pool config + prompts -> images (K per prompt) -> scores.csv
          build_pairs.py        pool scores + axis config -> pairs_v2/preferences.jsonl (CPU)
          build_selection.py    pool scores -> random / sel_aes / sel_tech survey set (CPU)
          run_ddpo.py           online RL (DDPO objective, Flow-GRPO SDE) -> lora_ddpo
          run_pipeline.py       config -> pairs -> DPO -> LoRA   (--pairs: train on pairs_v2)
          prompts_from_catalog.py  178 catalogue prompts -> prompts.txt
tests/    smoke_tiny.py         CPU test of the DPO step on a tiny random SD3 transformer
          smoke_ddpo_tiny.py    CPU test of the SDE log-prob + PPO loop (reward must rise) + identity wiring
          test_pool_pairs.py    CPU test of pool / legacy import / scoring cache / pairing
configs/  pool.yaml (shared pool), aesthetic.yaml, technical.yaml (+ pairs_v2 block), ddpo_*.yaml
```

## Shared preference pool (v2) — growing the DPO data to ≥ 2500 images

The first runs (Aug 2026) generated K=8 samples per prompt **separately for each
axis** with the same seeds (1000–1007), so `rl_aes/pairs/images` and
`rl_tech/pairs/images` are byte-identical: 1424 images and 356 pairs per axis.
v2 keeps one **shared pool** scored on both axes with K=16
(178 × 16 = **2848 images**, seeds 1000–1015) and pairs per axis from the cache.
v2 is aligned with the **RufGen survey grid** (`IAA_Bias_05_Guitar_Aesthetics_2500.ipynb`,
the 1350 plain images + aesthetics tables): LoRA `merged_finetune/rw` at scale 1.0,
the manifest prompt strings (trigger `Ruf Guitar Schrodinger 6`, template
`..., a single-cutaway electric guitar body, {color} finish, {top}, product photo on a
neutral gray background, studio lighting, sharp focus, three-quarter view, full body visible`),
**guidance 4.5**, 28 steps, 1024², CPU generator. v1 used the plain LoRA, the catalogue
prompt/token and guidance 7.0, so v1 images and adapters are not reused. Outputs go to
`Dokumetacja/reward-GENERATED/Dataset/RufGen/{pool,pairs_v2,generated/generated_rl_*_v2}`;
the image layout is `generated/<method>/SD35/{color}/{top}/seed_XXXX.png` with
`<method>` = `plain` (the 1350-image grid), `generated_rl_aes` / `generated_rl_tech` (v1), `*_v2`:

```bash
python scripts/build_pool.py  --config configs/pool.yaml --limit 2   # smoke (prompts: config `manifest`)
python scripts/build_pool.py  --config configs/pool.yaml             # resumable
# prompt sources: --prompts file | --manifest guitar_manifest.csv | --catalog combinations.json
# (catalogue prompts, token rewritten) | config manifest | existing {pool_dir}/prompts.txt
python scripts/build_pairs.py --config configs/aesthetic.yaml       # -> RufGen/pairs_v2/rl_aes/
python scripts/build_pairs.py --config configs/technical.yaml       # -> RufGen/pairs_v2/rl_tech/
python scripts/run_pipeline.py --config configs/aesthetic.yaml \
    --pairs .../RufGen/pairs_v2/rl_aes/preferences.jsonl --run-name lora_v2
python eval/generate_triplets.py --method rl_aes --catalog combinations.json --manifest guitar_manifest.csv \
    --plain-lora .../merged_finetune/rw --dpo-lora .../rl_aes/lora_v2 --out .../generated/generated_rl_aes_v2 \
    --seeds 0-6 --guidance 4.5
```

* `build_pool` generates only the missing `p{idx}_g{k}.png` and freezes `base_seed` /
  sampler settings in `pool/meta.json` (K may grow; prompts may not change).
  If you prefer to keep the catalogue token and reuse the 1424 v1 images, list
  `rl_aes/pairs` under `legacy_image_dirs` and build `prompts_pool.txt` from the
  prompt column of `rl_aes/pairs/scores.csv` (catalogue order); the import matches
  by prompt string.
* `pool/scores.csv` holds the raw metrics per image (4 by default) and is incremental.
* `build_pairs` standardises within each prompt's K samples (robust-z, as in
  the paper) and writes `preferences.jsonl` + `scores.csv` + `summary.json`.
  Strategies: `top_bottom` (ladder, ≤ K/2 pairs/prompt — the paper's pairing
  with more rungs: ≤ 1424 pairs at K=16), `ranked_gap` (rank gap ≥ g, strongest
  margins first, ≤ 2848 pairs at 16/prompt). On the legacy K=8 scores
  `top_bottom --max-pairs 2 --margin 0.1` reproduces the 356 pairs of each axis.
* `summary.json` reports `degenerate_mad_prompts`: prompts where a metric is
  constant over K, which makes its robust-z explode (this is what CLIP-IQA-sharp
  did on the v1 pool and why it was dropped).

Notebook: `sd35_dpo_pool_extend.ipynb` (Colab A100): `generated_v2/plain` (reference,
no DPO needed) → pool → pairs → `lora_v2` → `reward_gain` → `generated_v2/generated_rl_{aes,tech}`.
The v2 survey set lives in `Dataset/RufGen/generated_v2/<method>/SD35/{color}/{top}/seed_XXXX.png`
(`generate_triplets.py --no-suffix --manifest …`, seeds = `available_seeds`, 1350 per method),
i.e. the catalogue / bucket layout, so the three methods differ only by the DPO adapter.

## Method (what `train/dpo.py` does)

Diffusion-DPO (Wallace et al. 2023) needs, for each image, a per-sample
denoising error under the policy and under a frozen reference. For SD-3.5 the
denoiser is flow matching, so the error is the velocity-prediction MSE:

```
x_t = (1 - σ) x_0 + σ ε            σ from FlowMatchEulerDiscreteScheduler (shift 3)
v̂   = MMDiT(x_t, t, text)          target v = ε - x_0
e   = mean (v̂ - v)²                one scalar per image
loss = -log σ( β · [ (e_ref^w - e_ref^l) - (e_pol^w - e_pol^l) ] )
```

Chosen and rejected share ε and t. Timesteps are logit-normal (SD-3 default).
The plain LoRA is **fused** into the transformer; the DPO adapter is a fresh
LoRA (rank 32 / α 64) on the joint-attention projections; disabling the adapter
gives the reference. So `plain` and `rl_*` differ only by the DPO adapter.
LoRA params are kept in fp32 (bf16 would swallow a 2.6e-6 update).

## Trigger token — read this first

The plain LoRA (`results/merged_finetune/plain`, Notebook1_merged) was trained with
`Ruf Guitar Schrodinger 6` (dataset captions, E1–E4 use the same string).
`tasks_catalog.json` uses `Ruf Guitars Schrödinger 6`. `prompts_from_catalog.py`
and `generate_triplets.py` rewrite the catalogue token to the training one; if the
existing `plain` images in the bucket were generated with the catalogue string,
regenerate them with `--method plain` so all three methods share the prompt.

## Quick start (Colab A100)

```bash
pip install -r requirements.txt
python tests/smoke_tiny.py                       # CPU, no downloads, ~1 min

python scripts/prompts_from_catalog.py /path/mos-eval/data/tasks_catalog.json prompts.txt

# 1) smoke: 3 prompts, 5 steps
head -3 prompts.txt > p3.txt
python scripts/run_pipeline.py --config configs/aesthetic.yaml --prompts p3.txt --max-steps 5

# 2) full runs
python scripts/run_pipeline.py --config configs/aesthetic.yaml --prompts prompts.txt
python scripts/run_pipeline.py --config configs/technical.yaml --prompts prompts.txt

# 3) survey images (same prompts+seeds as the catalogue)
# writes seed_XXXX_aes.png / seed_XXXX_tech.png next to the plain seed_XXXX.png
python eval/generate_triplets.py --method rl_aes  --catalog tasks_catalog.json \
    --plain-lora .../merged_finetune/plain --dpo-lora .../rl_aes/lora  --out .../RufGen/generated
python eval/generate_triplets.py --method rl_tech --catalog tasks_catalog.json \
    --plain-lora .../merged_finetune/plain --dpo-lora .../rl_tech/lora --out .../RufGen/generated
gsutil -m rsync -r .../RufGen/generated/SD35 gs://ruf-ai/rufgen/generated/SD35
```

Set `--steps/--guidance` in `generate_triplets.py` to whatever produced the
existing `plain` images, otherwise the comparison is confounded by the sampler.

## From a notebook

```python
import sys; sys.path.insert(0, "/content/sd35-dpo")
from reward import RewardModel
from train.dpo import DPOConfig, build_policy, train_dpo, load_policy_pipeline
from train.data import PreferenceDataset
```

## Two ways out of "the triplets look identical"

### A. Reward-guided selection (no training) — `sd35_selection_survey.ipynb`
`scripts/build_selection.py` turns the scored pool into a survey set: per prompt one
**random** seed, the seed the **aesthetic** composite ranks first (`sel_aes`) and the
seed the **technical** composite ranks first (`sel_tech`); three different images,
`generated_v2_sel/{random,sel_aes,sel_tech}/SD35/{color}/{top}/seed_XXXX.png` +
`selection.csv` / `selection_summary.json`. The mos-eval catalogue builder has a
`--selection` mode (different seed per arm, `meta.selection=true`; the samplers keep
those seeds). This tests directly whether people see what the predictors see.

### B. DDPO / online RL — `train/ddpo.py`, `scripts/run_ddpo.py`, `sd35_ddpo.ipynb`
DDPO's PPO objective on per-step log-probabilities, made possible on rectified flow by
the **Flow-GRPO SDE sampler** (same marginals as the ODE, Gaussian steps). Per epoch:
K=8 rollouts × 8 prompts at 768²/T=10, reward = the DPO reward model, advantage =
within-group standardisation, PPO-clip (1e-4 on per-dim mean log-ratio), LR 1e-4 on the
same LoRA adapter, optional KL to the reference (`beta_kl`). `tests/smoke_ddpo_tiny.py`
checks log-prob consistency and that PPO raises a synthetic reward on a tiny SD3
transformer (CPU). Configs: `configs/ddpo_{aesthetic,technical}.yaml`; output
`{out_dir}/lora_ddpo` loads with `generate_triplets.py --dpo-lora` like a DPO adapter.
Watch `reward_mean` per epoch in `train_log.json`; stop at the plateau / before the
images look over-processed.

**Identity term** (`reward/identity.py`, after the first DDPO run drifted — maple neck,
tremolo arm, cables: the predictors know nothing about the product):
`r = z_group(quality) + identity_lam · min(0, (clip_sim − τ)/σ)`, where `clip_sim` is the
mean top-5 cosine similarity (CLIP ViT-L/14) to the 50 reference photos and τ/σ are
calibrated on the untuned model's samples, so only falling *below* the plain identity
level is penalised. Keep the advantage group-relative (`adv_mode: group`): a running
baseline (`global`) was tried on 2026-09-07 and collapsed — as soon as the shape penalty
kicks in every advantage is negative, PPO lowers the probability of everything it sampled
and the policy drifts in a random direction (shape 1.0 → 0.30 in 4 epochs). `eval/identity.py` scores folders
(plain vs adapters, same prompt + seed) and rollouts; notebook sections 6b/6c compare
checkpoints and report the identity cost.

**Shape term** (`reward/shape.py`) — the geometry must not move. With `shape_lam > 0`
every rollout is paired with a rollout of the *reference* model from the same initial
and per-step noise (adapter disabled, no grad), and the multi-scale Pearson correlation
of the Sobel edge maps of the pair (`shape`) enters the reward as
`shape_lam · min(0, shape − (1 − shape_tol))`. Calibration on synthetic scenes: colour /
texture change ≈ 1.0, hardware removed 0.92, 4 px shift 0.81, 8 px 0.56, 10 % zoom 0.26,
unrelated images 0.0. Defaults `shape_lam 5`, `shape_tol 0.05`: a 10 % zoom costs ≈ 3.5 σ
of quality reward. Extra cost ≈ +40 % epoch time.

## If the triplets look identical (they did with β=100)

`implicit_acc` climbing to 0.6–0.7 means the policy learns the right direction, but with
β=100 / LR 2.56e-6 / 500 steps the shift is far below what a viewer notices. Measure before
generating survey images (`eval/reward_gain.py` → `report.json`: mean gain + win-rate on the
adapter's own axis, held-out seeds 100+), then strengthen:

* `--dpo-scale 2` (generate_triplets / reward_gain): adapter strength at inference, no training;
* `run_pipeline.py --init-lora .../lora_v2 --max-steps 1500`: continue from a saved adapter;
* `run_pipeline.py --beta 10 --lr 1e-5 --max-steps 1000 --run-name lora_v2_b10`: weaker KL anchor.

Notebook section 8 runs all three and tabulates the reports.

## What to watch in the log

`implicit_acc` = share of pairs where the policy already prefers the chosen
image more than the reference does; it should climb from ~0.5. `pol_w` should
fall below `ref_w` while `pol_l` rises above `ref_l`. If the loss drops to ~0
within a few dozen steps, β is too large for this data — try 10–50.

## Budget (SD-3.5-medium, 1024², A100 40 GB)

Pair generation: 178 prompts × 8 samples ≈ 1 400 images ≈ 2–3 h.
Pool v2: 2 848 images (seeds 1000–1015, training token) ≈ 4.5–5 h, resumable; scoring 2 848 images × 4 metrics ≈ 15–25 min; pairing: seconds (CPU).
DPO: 500 steps × 4 pairs ≈ 2 000 pair-forwards (policy with grad + reference) ≈ 2–4 h.
Triplets: 178 × 7 seeds × 2 methods ≈ 2 500 images ≈ 4–5 h. Save to Drive as you go.

## Status

* `reward/`, `data/`, `train/`, `eval/`: implemented. `tests/smoke_tiny.py` exercises
  the full DPO step (LoRA attach, adapter on/off reference, loss, gradients, one
  optimiser step) on a tiny random SD3 transformer.
* Full v1 run done (Aug 26 2026, `notebooks/sd35_dpo_pipeline_v1_RUN_2026-08-26.ipynb`): 356 pairs/axis,
  500 DPO steps per axis (`rl_aes/lora`, `rl_tech/lora`), 1246 survey images per method in
  `RufGen/generated/generated_rl_{aes,tech}` — generated with the plain LoRA, catalogue prompts and
  guidance 7.0, i.e. NOT comparable with the plain grid (rw LoRA, manifest prompts, guidance 4.5).
* v2 shared pool: code + `tests/test_pool_pairs.py` (CPU, passes) + regression of
  `build_pairs` against the v1 `preferences.jsonl`; generation/scoring NOT yet run on GPU —
  run `sd35_dpo_pool_extend.ipynb` (smoke `--limit 2` first).
* Hyper-parameters (rank 32/α 64, LR 2.56e-6, β 100, 500 steps) come from the
  handoff (arXiv:2605.19839); β is the least certain — see "What to watch".
