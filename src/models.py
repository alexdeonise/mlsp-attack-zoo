"""Incarca un model timm si il impacheteaza astfel incat:
  - primeste imagini in [0,1] (normalizarea ImageNet e inauntru)
  - scoate doar logit-urile claselor care ne intereseaza (10 pentru Imagenette)
Asa atacurile din torchattacks lucreaza corect, in [0,1]."""
import os, torch, torch.nn as nn, timm
from data import IMAGENETTE_TO_IMAGENET

NAMES = {
    "resnet50":     "resnet50.a1_in1k",
    "efficientnet": "efficientnet_b0.ra_in1k",
    "vit":          "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "convnext":     "convnext_tiny.fb_in1k",
    # checkpoints that differ only in pretraining data (camera-ready, reviewer 2)
    "vit_in1k":       "vit_base_patch16_224.augreg_in1k",   # same ViT and recipe, no ImageNet-21k
    "convnext_in22k": "convnext_tiny.fb_in22k_ft_in1k",     # same ConvNeXt, pretrained on ImageNet-22k
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

class Wrapped(nn.Module):
    def __init__(self, backbone, class_idx=None):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", MEAN); self.register_buffer("std", STD)
        self.class_idx = class_idx
    def forward(self, x):
        out = self.backbone((x - self.mean) / self.std)
        return out[:, self.class_idx] if self.class_idx is not None else out

def load(short: str, dataset: str, device="cuda"):
    """dataset='imagenette' -> preantrenat ImageNet, selectam 10 clase.
       dataset='cifar10'    -> incarcam checkpoint-ul din finetune_cifar.py."""
    m = timm.create_model(NAMES[short], pretrained=(dataset == "imagenette"), num_classes=1000 if dataset=="imagenette" else 10)
    if dataset == "cifar10":
        ck = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{short}_cifar10.pt")
        m.load_state_dict(torch.load(ck, map_location="cpu"))
    idx = torch.tensor(IMAGENETTE_TO_IMAGENET) if dataset == "imagenette" else None
    return Wrapped(m, idx).to(device).eval()
