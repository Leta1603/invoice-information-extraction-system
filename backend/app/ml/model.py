"""FCOS document detector model."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    """Conv2d → BatchNorm2d → ReLU building block."""

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


class _Backbone(nn.Module):
    """Lightweight CNN backbone producing multi-scale feature maps C3/C4/C5."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.ModuleList([
            ConvBN(3, 32, k=3, s=2, p=1),
            ConvBN(32, 32, k=3, s=1, p=1),
        ])
        self.stage2 = nn.ModuleList([
            ConvBN(32, 64, k=3, s=2, p=1),
            ConvBN(64, 64, k=3, s=1, p=1),
        ])
        self.stage3 = nn.ModuleList([
            ConvBN(64, 128, k=3, s=2, p=1),
            ConvBN(128, 128, k=3, s=1, p=1),
        ])
        self.stage4 = nn.ModuleList([
            ConvBN(128, 256, k=3, s=2, p=1),
            ConvBN(256, 256, k=3, s=1, p=1),
        ])
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
        c3 = x
        for block in self.stage4:
            x = block(x)
        c4 = x
        for block in self.stage5:
            x = block(x)
        c5 = x
        return c3, c4, c5


class _FPN(nn.Module):
    """Feature Pyramid Network with 3 levels (P3/P4/P5) and 128 output channels."""

    def __init__(self) -> None:
        super().__init__()
        self.lat3 = nn.Conv2d(128, 128, 1)
        self.lat4 = nn.Conv2d(256, 128, 1)
        self.lat5 = nn.Conv2d(512, 128, 1)
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


class _Head(nn.Module):
    """Shared detection head for all FPN levels."""

    def __init__(self, in_ch: int = 128, num_classes: int = 5) -> None:
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


class FCOSDocumentDetector(nn.Module):
    """FCOS detector for invoice field localisation."""

    NUM_CLASSES: int = 5
    CLASS_NAMES: List[str] = [
        "seller",
        "invoice_number",
        "invoice_date",
        "total_amount",
        "item_description",
    ]
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
