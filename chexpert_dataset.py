"""CheXpert dataset for the hybrid CBM: each sample needs BOTH the 5 target-pathology
labels (classification target) AND 9 concept labels (the other CheXpert findings --
same idea as baseline/post-hoc-cbm/data/constants.py's CHEXPERT_CONCEPT_COLUMNS: no
separate concept dataset exists for chest X-rays, so co-observed findings that aren't
themselves predicted serve as concepts, mirroring how MONET concepts are separate from
MILK10k's isicdx diagnosis labels)."""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_DATA_DIR = "/scratch2/[SC_LAB]/dataset/causal/chexpert"
PATHOLOGIES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
CONCEPT_COLUMNS = [
    "No Finding", "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion",
    "Pneumonia", "Pneumothorax", "Pleural Other", "Fracture", "Support Devices",
]


def resolve_image_path(data_dir: str, csv_path: str) -> str:
    rel = csv_path.lstrip("./")
    for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return os.path.join(data_dir, rel)


class CheXpertCBMDataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str = DEFAULT_DATA_DIR, transform=None):
        df = pd.read_csv(csv_path)
        for col in PATHOLOGIES + CONCEPT_COLUMNS:
            df[col] = df[col].fillna(0.0).replace(-1.0, 1.0)  # U-Ones for uncertain labels
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(resolve_image_path(self.data_dir, row["Path"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        diagnosis = torch.tensor([row[p] for p in PATHOLOGIES], dtype=torch.float32)
        concepts = torch.tensor([row[c] for c in CONCEPT_COLUMNS], dtype=torch.float32)
        return img, diagnosis, concepts
