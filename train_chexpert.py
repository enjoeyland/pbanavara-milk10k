#!/usr/bin/env python3
"""Two-phase finetune of the hybrid CBM (chexpert_model.SingleImageHybridCBM) on
CheXpert, mirroring src/train.py's own recipe: Phase 1 freezes DINOv2 and trains only
the heads (fast warm-up), Phase 2 unfreezes the backbone with a much smaller LR. The
manuscript found this ordering necessary for MILK10k (unfreezing too early / with too
high an LR regressed the concept head -- see baselines_scores.xlsx notes); we keep the
same shape here, just compressed to CheXpert's much larger (223k image) train set.

Joint loss: BCE(concepts) + BCE(classification), both multi-label/multi-binary here
(src/train.py used MSE for concepts since MONET scores are continuous 0-1; our
concepts are the binary/uncertain-mapped CheXpert findings, so BCE is the right fit).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from chexpert_dataset import CONCEPT_COLUMNS, DEFAULT_DATA_DIR, PATHOLOGIES, CheXpertCBMDataset
from chexpert_model import SingleImageHybridCBM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics_util  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--phase1-epochs", type=int, default=1)
    p.add_argument("--phase2-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--phase1-lr", type=float, default=1e-3)
    p.add_argument("--phase2-backbone-lr", type=float, default=2e-6)
    p.add_argument("--phase2-head-lr", type=float, default=1e-4)
    p.add_argument("--concept-loss-weight", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--output-dir", default="outputs_chexpert")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, _ in tqdm(loader, desc="eval"):
        probs = torch.sigmoid(model(imgs.to(device))["logits"]).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def run_epoch(model, loader, optimizer, scaler, cls_criterion, concept_criterion, concept_weight, device, desc):
    model.train()
    running_loss = 0.0
    for imgs, labels, concepts in tqdm(loader, desc=desc):
        imgs, labels, concepts = imgs.to(device), labels.to(device), concepts.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(imgs)
        # concept_head ends in Sigmoid, and BCELoss (unlike BCEWithLogitsLoss) is not
        # autocast-safe -- cast back to float32 before computing it.
        loss = (
            concept_weight * concept_criterion(out["concepts"].float(), concepts)
            + cls_criterion(out["logits"].float(), labels)
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = CheXpertCBMDataset(f"{args.data_dir}/train.csv", args.data_dir, train_transform)
    if args.limit_train > 0:
        train_ds.df = train_ds.df.iloc[: args.limit_train].reset_index(drop=True)
    val_ds = CheXpertCBMDataset(f"{args.data_dir}/valid.csv", args.data_dir, val_transform)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}, concepts: {CONCEPT_COLUMNS}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SingleImageHybridCBM(num_concepts=len(CONCEPT_COLUMNS), num_classes=len(PATHOLOGIES)).to(device)
    cls_criterion = nn.BCEWithLogitsLoss()
    concept_criterion = nn.BCELoss()  # concept_head already ends in Sigmoid
    scaler = torch.cuda.amp.GradScaler()

    best_mean_auc = -1.0

    def eval_and_maybe_save(tag):
        nonlocal best_mean_auc
        probs, labels = evaluate(model, val_loader, device)
        per_path = {}
        for i, name in enumerate(PATHOLOGIES):
            if len(np.unique(labels[:, i])) > 1:
                per_path[name] = metrics_util.compute_binary_metrics(labels[:, i].astype(int), probs[:, i])
        mean_metrics = metrics_util.average_binary_metrics(per_path) if per_path else {}
        mean_auc = mean_metrics.get("roc_auc") or -1.0
        print(f"{tag}: val mean_auc={mean_auc:.4f} mean_metrics={mean_metrics}")
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            torch.save({"model": model.state_dict()}, out_dir / "checkpoint_best.pth")
            pred_df = pd.DataFrame({"Path": val_ds.df["Path"]})
            for i, name in enumerate(PATHOLOGIES):
                pred_df[f"prob_{name}"] = probs[:, i]
                pred_df[f"label_{name}"] = labels[:, i]
            pred_df.to_csv(out_dir / "val_predictions.csv", index=False)

    print("=== Phase 1: frozen backbone ===")
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.phase1_lr, weight_decay=0.05
    )
    for epoch in range(args.phase1_epochs):
        t0 = time.time()
        loss = run_epoch(model, train_loader, optimizer, scaler, cls_criterion, concept_criterion,
                          args.concept_loss_weight, device, f"phase1 epoch {epoch}")
        print(f"phase1 epoch {epoch}: loss={loss:.4f} time={time.time() - t0:.0f}s")
        eval_and_maybe_save(f"phase1 epoch {epoch}")

    print("=== Phase 2: full finetune (differential LR) ===")
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.phase2_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": args.phase2_head_lr},
    ], weight_decay=0.05)
    for epoch in range(args.phase2_epochs):
        t0 = time.time()
        loss = run_epoch(model, train_loader, optimizer, scaler, cls_criterion, concept_criterion,
                          args.concept_loss_weight, device, f"phase2 epoch {epoch}")
        print(f"phase2 epoch {epoch}: loss={loss:.4f} time={time.time() - t0:.0f}s")
        eval_and_maybe_save(f"phase2 epoch {epoch}")

    print(f"done. best mean AUC={best_mean_auc:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
