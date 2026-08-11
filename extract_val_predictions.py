#!/usr/bin/env python3
"""Run outputs/best_model.pt on its own val split and save predictions to CSV.

No prediction CSV was saved during training (only the checkpoint + split_indices.json),
so this reconstructs the identical val split via src.dataset.get_dataloaders (same
config, same seed) and re-runs inference once. Output feeds baseline/metrics_util.py.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import DIAGNOSIS_COLUMNS, get_dataloaders
from src.model import get_model
from src.utils import get_device, load_checkpoint, load_config

import pandas as pd


def main():
    config = load_config("configs/default.yaml")
    device = get_device()

    _, val_loader, _ = get_dataloaders(config)

    model = get_model(config).to(device)
    load_checkpoint("outputs/best_model.pt", model)
    model.eval()

    lesion_ids, true_labels, probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="inference"):
            clinical = batch["clinical_image"].to(device)
            dermoscopic = batch["dermoscopic_image"].to(device)
            outputs = model(clinical, dermoscopic)
            probs.append(F.softmax(outputs["logits"], dim=1).cpu())
            true_labels.append(batch["diagnosis"].argmax(dim=1))
            lesion_ids.extend(batch["lesion_id"])

    probs = torch.cat(probs).numpy()
    true_labels = torch.cat(true_labels).numpy()
    pred_labels = probs.argmax(axis=1)

    df = pd.DataFrame({"lesion_id": lesion_ids, "true_label": true_labels, "pred_label": pred_labels})
    for i, name in enumerate(DIAGNOSIS_COLUMNS):
        df[f"prob_{i}"] = probs[:, i]
    df.to_csv("outputs/val_predictions.csv", index=False)
    print(f"Wrote outputs/val_predictions.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
