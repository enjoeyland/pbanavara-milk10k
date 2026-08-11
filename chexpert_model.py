"""Single-image variant of src/model.py's hybrid Concept Bottleneck Model, for CheXpert.

src/model.py's ConceptBottleneckModel is hard-wired for two co-registered images
(clinical + dermoscopic) fused via concat before the concept/classification heads --
CheXpert has one frontal/lateral view per study, not a paired structure, so this drops
the second backbone call and halves the fusion layer's input width. Same
concept-bottleneck + residual-path idea otherwise (see src/model.py's docstring for the
strict/hybrid/baseline variant explanation); only "hybrid" is implemented here since
that's what won on MILK10k (macro F1 0.529 vs 0.169 for post-hoc's linear-only
equivalent -- see baseline/model_metrics.csv).
"""
import torch
import torch.nn as nn


class SingleImageHybridCBM(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov2_vitl14",
        backbone_dim: int = 1024,
        fusion_dim: int = 512,
        num_concepts: int = 9,
        num_classes: int = 5,
        residual_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=True)
        self.freeze_backbone()

        self.fusion = nn.Sequential(nn.Linear(backbone_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.concept_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_concepts), nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(nn.Linear(fusion_dim, residual_dim), nn.Dropout(dropout))
        self.classification_head = nn.Sequential(
            nn.Linear(num_concepts + residual_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, image: torch.Tensor) -> dict:
        cls_tok = self.backbone(image)
        fused = self.fusion(cls_tok)
        concepts = self.concept_head(fused)
        residual = self.residual_head(fused)
        logits = self.classification_head(torch.cat([concepts, residual], dim=1))
        return {"logits": logits, "concepts": concepts}
