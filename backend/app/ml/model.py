"""
FCOSDocumentDetector — architecture matches best_fcos_detector.pt exactly.

Checkpoint key structure (114 tensors):
  backbone.stem.{0,1}.block.{0=Conv,1=BN}
  backbone.stage{2..5}.{0,1}.block.{0=Conv,1=BN}
  fpn.lat{3,4,5}.{weight,bias}          — plain Conv2d (1x1)
  fpn.out{3,4,5}.block.{0=Conv,1=BN}   — ConvBN (3x3)
  head.shared.{0..3}.block.{0=Conv,1=BN}
  head.cls_logits / head.ctr_logits / head.bbox_reg — plain Conv2d (3x3)

Channel progression:
  stem      : 3 → 32 (2 ConvBN, stride-2 then stride-1)
  stage2    : 32 → 64
  stage3    : 64 → 128   → C3  stride-8
  stage4    : 128 → 256  → C4  stride-16
  stage5    : 256 → 512  → C5  stride-32
  FPN out   : 128 channels, 3 levels  strides [8, 16, 32]
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block: Conv + BN + ReLU stored as self.block (sequential)
# so state-dict paths are  <name>.block.0.*  and  <name>.block.1.*
# ---------------------------------------------------------------------------

class ConvBN(nn.Module):
    """Conv2d → BatchNorm2d → ReLU building block.

    State-dict paths for weights are stored as
    ``<name>.block.0.*`` (Conv2d) and ``<name>.block.1.*`` (BN)
    to match the training checkpoint layout.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        """Initialise convolutional block.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            k: Kernel size.
            s: Stride.
            p: Padding.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv → BN → ReLU to *x*."""
        return self.block(x)


# ---------------------------------------------------------------------------
# Backbone — custom lightweight CNN
# ---------------------------------------------------------------------------

class _Backbone(nn.Module):
    """Lightweight CNN backbone producing multi-scale feature maps C3/C4/C5."""

    def __init__(self) -> None:
        super().__init__()
        # stem: two ConvBN blocks (stride-2 → stride-1)
        self.stem = nn.ModuleList([
            ConvBN(3, 32, k=3, s=2, p=1),    # stem.0
            ConvBN(32, 32, k=3, s=1, p=1),   # stem.1
        ])
        # stage2: stride-4
        self.stage2 = nn.ModuleList([
            ConvBN(32, 64, k=3, s=2, p=1),
            ConvBN(64, 64, k=3, s=1, p=1),
        ])
        # stage3: stride-8  → C3 (128 ch)
        self.stage3 = nn.ModuleList([
            ConvBN(64, 128, k=3, s=2, p=1),
            ConvBN(128, 128, k=3, s=1, p=1),
        ])
        # stage4: stride-16 → C4 (256 ch)
        self.stage4 = nn.ModuleList([
            ConvBN(128, 256, k=3, s=2, p=1),
            ConvBN(256, 256, k=3, s=1, p=1),
        ])
        # stage5: stride-32 → C5 (512 ch)
        self.stage5 = nn.ModuleList([
            ConvBN(256, 512, k=3, s=2, p=1),
            ConvBN(512, 512, k=3, s=1, p=1),
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass and return (C3, C4, C5) feature tensors.

        Args:
            x: Input image batch of shape ``[B, 3, H, W]``.

        Returns:
            Tuple of (C3, C4, C5) tensors with strides 8, 16, 32.
        """
        for block in self.stem:
            x = block(x)
        for block in self.stage2:
            x = block(x)
        for block in self.stage3:
            x = block(x)
        c3 = x  # 128 ch, stride 8
        for block in self.stage4:
            x = block(x)
        c4 = x  # 256 ch, stride 16
        for block in self.stage5:
            x = block(x)
        c5 = x  # 512 ch, stride 32
        return c3, c4, c5


# ---------------------------------------------------------------------------
# FPN — 3 levels (P3/P4/P5), 128 channels
# ---------------------------------------------------------------------------

class _FPN(nn.Module):
    """Feature Pyramid Network with 3 levels (P3/P4/P5) and 128 output channels."""

    def __init__(self) -> None:
        super().__init__()
        # plain 1x1 lateral convs (with bias → stored as .weight / .bias directly)
        self.lat3 = nn.Conv2d(128, 128, 1)
        self.lat4 = nn.Conv2d(256, 128, 1)
        self.lat5 = nn.Conv2d(512, 128, 1)
        # output ConvBN (3x3)
        self.out3 = ConvBN(128, 128, k=3, s=1, p=1)
        self.out4 = ConvBN(128, 128, k=3, s=1, p=1)
        self.out5 = ConvBN(128, 128, k=3, s=1, p=1)

    def forward(
        self,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Build feature pyramid from backbone outputs.

        Args:
            c3: Backbone C3 feature map (stride 8, 128 ch).
            c4: Backbone C4 feature map (stride 16, 256 ch).
            c5: Backbone C5 feature map (stride 32, 512 ch).

        Returns:
            List of [P3, P4, P5] tensors, each with 128 channels.
        """
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return [self.out3(p3), self.out4(p4), self.out5(p5)]


# ---------------------------------------------------------------------------
# Shared detection head — processes one feature map at a time (like notebook)
# ---------------------------------------------------------------------------

class _Head(nn.Module):
    """Shared detection head applied to each FPN level.

    Outputs raw (un-sigmoided) logits for class, centerness, and bbox regression.
    """

    def __init__(self, in_ch: int = 128, num_classes: int = 5) -> None:
        """Initialise detection head.

        Args:
            in_ch: Number of input feature channels (must match FPN output).
            num_classes: Number of object classes to detect.
        """
        super().__init__()
        self.shared = nn.ModuleList([ConvBN(in_ch, in_ch) for _ in range(4)])
        self.cls_logits = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.ctr_logits = nn.Conv2d(in_ch, 1, 3, padding=1)
        self.bbox_reg = nn.Conv2d(in_ch, 4, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """Compute classification, centerness, and box regression outputs.

        Args:
            x: Feature map tensor of shape ``[B, C, H, W]``.

        Returns:
            Tuple of (cls_logits, ctr_logits, bbox_reg) tensors.
        """
        for block in self.shared:
            x = block(x)
        cls_logits = self.cls_logits(x)
        ctr_logits = self.ctr_logits(x)
        bbox_reg = F.relu(self.bbox_reg(x))
        return cls_logits, ctr_logits, bbox_reg


# ---------------------------------------------------------------------------
# Public detector class
# ---------------------------------------------------------------------------

class FCOSDocumentDetector(nn.Module):
    """FCOS-based detector for invoice field localisation.

    Combines a lightweight CNN backbone, a 3-level FPN, and a shared
    detection head.  Architecture matches ``best_fcos_detector.pt`` exactly
    (114 state-dict keys).
    """

    NUM_CLASSES: int = 5
    CLASS_NAMES: List[str] = [
        "seller",
        "invoice_number",
        "invoice_date",
        "total_amount",
        "item_description",
    ]
    # 3 FPN levels only
    FPN_STRIDES: List[int] = [8, 16, 32]

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.fpn = _FPN()
        self.head = _Head(in_ch=128, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> List[dict]:
        """Run detection on a batch of images.

        Args:
            images: Batch tensor of shape ``[B, 3, H, W]``, normalised
                with mean=0.5 and std=0.5.

        Returns:
            List of per-level output dicts, each containing
            ``level``, ``stride``, ``cls_logits``, ``ctr_logits``, ``bbox_reg``.
        """
        c3, c4, c5 = self.backbone(images)
        features = self.fpn(c3, c4, c5)

        outputs = []
        for level_name, feat, stride in zip(
            ["p3", "p4", "p5"], features, self.FPN_STRIDES
        ):
            cls_logits, ctr_logits, bbox_reg = self.head(feat)
            outputs.append({
                "level": level_name,
                "stride": stride,
                "cls_logits": cls_logits,
                "ctr_logits": ctr_logits,
                "bbox_reg": bbox_reg,
            })
        return outputs


