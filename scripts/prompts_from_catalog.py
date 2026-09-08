#!/usr/bin/env python3
"""Extract the unique prompts of tasks_catalog.json into a prompts file for pair
generation, rewriting the catalogue token to the TRAINING trigger.

The catalogue was written with "Ruf Guitars Schrödinger 6", but every fine-tuning
notebook (Notebook1_merged, E1-E4) and the dataset captions use
"Ruf Guitar Schrodinger 6" — the LoRA only knows the latter.

    python scripts/prompts_from_catalog.py tasks_catalog.json prompts.txt
"""
import json, sys
CATALOG_TOKEN = "Ruf Guitars Schr\u00f6dinger 6"
TRAIN_TOKEN = "Ruf Guitar Schrodinger 6"
cat, out = sys.argv[1], sys.argv[2]
tasks = json.load(open(cat, encoding="utf-8"))["tasks"]
prompts = sorted({t["prompt"].replace(CATALOG_TOKEN, TRAIN_TOKEN) for t in tasks})
bad = [p for p in prompts if TRAIN_TOKEN not in p]
if bad:
    print(f"WARNING: {len(bad)} prompts without the training token, e.g. {bad[0][:60]}")
open(out, "w", encoding="utf-8").write("\n".join(prompts) + "\n")
print(f"{len(prompts)} prompts -> {out} (token: {TRAIN_TOKEN!r})")
