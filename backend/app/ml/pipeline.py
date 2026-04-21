
"""
Full inference pipeline — exact port of the training notebook logic.

Key differences vs. a generic FCOS pipeline:
  * Image is resized to a fixed 512×512 (no aspect-ratio preservation)
  * Normalisation: mean=0.5, std=0.5  (NOT ImageNet)
  * Score = 0.7·cls_prob + 0.3·ctr_prob
  * Per-class spatial heuristics for postprocessing
  * Box expansion before OCR
  * Per-field OCR config (PSM, whitelists) and cleaning
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.ops import nms

from app.core.config import settings
from app.ml.load_model import load_model
from app.ml.model import FCOSDocumentDetector

logger = logging.getLogger(__name__)

IMG_SIZE = 512

# Class IDs (must match training label2id)
SELLER_ID = 0
INVOICE_NUMBER_ID = 1
INVOICE_DATE_ID = 2
TOTAL_AMOUNT_ID = 3
ITEM_DESCRIPTION_ID = 4

ID2LABEL = {
    0: "seller",
    1: "invoice_number",
    2: "invoice_date",
    3: "total_amount",
    4: "item_description",
}

# ═══════════════════════════════════════════════════════════════════════════
# 1. Image preprocessing — matches notebook InvoiceDataset exactly
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize to 512×512, normalize with mean=0.5/std=0.5. Returns [1,3,512,512]."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).permute(2, 0, 1)  # [3,H,W]
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor


# ═══════════════════════════════════════════════════════════════════════════
# 2. FCOS decoding — matches notebook decode_fcos_predictions
# ═══════════════════════════════════════════════════════════════════════════

def _get_feature_map_locations(h: int, w: int, stride: int, device):
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # [H, W, 2]


def decode_fcos_predictions(
    outputs: List[dict],
    img_size: int = IMG_SIZE,
    score_threshold: float = 0.2,
    iou_threshold: float = 0.5,
    top_k: int = 30,
) -> list:
    """
    Vectorised version of the notebook's decode.
    Score = 0.7·class_prob + 0.3·centerness_prob.
    Returns list of [x1, y1, x2, y2, score, class_id].
    """
    all_boxes, all_scores, all_labels = [], [], []

    for out in outputs:
        stride = out["stride"]
        cls_probs = torch.sigmoid(out["cls_logits"])[0]       # [C, H, W]
        ctr_probs = torch.sigmoid(out["ctr_logits"])[0, 0]    # [H, W]
        reg_pred = out["bbox_reg"][0]                          # [4, H, W]

        C, h, w = cls_probs.shape
        locations = _get_feature_map_locations(h, w, stride, cls_probs.device)  # [H,W,2]

        # Best class per location
        class_scores, class_ids = cls_probs.max(dim=0)  # both [H, W]

        # Combined score (notebook formula)
        combined = 0.7 * class_scores + 0.3 * ctr_probs  # [H, W]

        mask = combined > score_threshold
        if mask.sum() == 0:
            continue

        px = locations[mask][:, 0]
        py = locations[mask][:, 1]
        l = reg_pred[0][mask]
        t = reg_pred[1][mask]
        r = reg_pred[2][mask]
        b = reg_pred[3][mask]

        x1 = (px - l).clamp(0, img_size)
        y1 = (py - t).clamp(0, img_size)
        x2 = (px + r).clamp(0, img_size)
        y2 = (py + b).clamp(0, img_size)

        valid = (x2 > x1) & (y2 > y1)
        if valid.sum() == 0:
            continue

        all_boxes.append(torch.stack([x1, y1, x2, y2], dim=1)[valid])
        all_scores.append(combined[mask][valid])
        all_labels.append(class_ids[mask][valid])

    if not all_boxes:
        return []

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # Per-class NMS (matching notebook)
    final_preds = []
    for cls_id in labels.unique():
        cls_mask = labels == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_threshold)
        for idx in keep:
            b = cls_boxes[idx]
            final_preds.append([
                float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                float(cls_scores[idx]),
                int(cls_id),
            ])

    final_preds.sort(key=lambda x: x[4], reverse=True)
    return final_preds[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Spatial postprocessing — exact port of notebook heuristics
# ═══════════════════════════════════════════════════════════════════════════

def _get_preds_by_class(preds, class_id):
    return [p for p in preds if p[5] == class_id]

def _box_area(p):
    return max(0, p[2] - p[0]) * max(0, p[3] - p[1])

def _box_center(p):
    return (p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0


def _pick_best_invoice_number(preds, img_w=IMG_SIZE, img_h=IMG_SIZE):
    candidates = _get_preds_by_class(preds, INVOICE_NUMBER_ID)
    if not candidates:
        return None
    filtered = []
    for p in candidates:
        xc, yc = _box_center(p)
        if yc > img_h * 0.35:
            continue
        if (p[2] - p[0]) > img_w * 0.45:
            continue
        filtered.append(p)
    if filtered:
        candidates = filtered
    return max(candidates, key=lambda p: p[4])


def _pick_best_invoice_date(preds, img_w=IMG_SIZE, img_h=IMG_SIZE):
    candidates = _get_preds_by_class(preds, INVOICE_DATE_ID)
    if not candidates:
        return None
    filtered = []
    for p in candidates:
        xc, yc = _box_center(p)
        if yc > img_h * 0.4:
            continue
        if (p[2] - p[0]) > img_w * 0.7:
            continue
        filtered.append(p)
    if filtered:
        candidates = filtered
    return max(candidates, key=lambda p: p[4])


def _pick_best_seller(preds, img_w=IMG_SIZE, img_h=IMG_SIZE):
    candidates = _get_preds_by_class(preds, SELLER_ID)
    if not candidates:
        return None
    filtered = []
    for p in candidates:
        xc, yc = _box_center(p)
        if yc > img_h * 0.55:
            continue
        if (p[3] - p[1]) < 25:
            continue
        filtered.append(p)
    if filtered:
        candidates = filtered

    def rank(p):
        xc, yc = _box_center(p)
        return 0.6 * p[4] + 0.25 * (1 - xc / img_w) + 0.15 * (1 - yc / img_h)

    return max(candidates, key=rank)


def _pick_best_item_description(preds, img_w=IMG_SIZE, img_h=IMG_SIZE):
    candidates = _get_preds_by_class(preds, ITEM_DESCRIPTION_ID)
    if not candidates:
        return None
    filtered = []
    for p in candidates:
        xc, yc = _box_center(p)
        w = p[2] - p[0]
        area = _box_area(p)
        if yc < img_h * 0.18 or yc > img_h * 0.72:
            continue
        if w < img_w * 0.35:
            continue
        if area > img_w * img_h * 0.45:
            continue
        filtered.append(p)
    if filtered:
        candidates = filtered

    def rank(p):
        w = p[2] - p[0]
        area = _box_area(p)
        return 0.55 * p[4] + 0.25 * (w / img_w) + 0.20 * min(area / (img_w * img_h * 0.25), 1.0)

    return max(candidates, key=rank)


def _pick_best_total_amount(preds, item_desc_box=None, img_w=IMG_SIZE, img_h=IMG_SIZE):
    candidates = _get_preds_by_class(preds, TOTAL_AMOUNT_ID)
    if not candidates:
        return None
    filtered = []
    for p in candidates:
        xc, yc = _box_center(p)
        if yc < img_h * 0.45:
            continue
        if xc < img_w * 0.45:
            continue
        if _box_area(p) > img_w * img_h * 0.12:
            continue
        if item_desc_box is not None and yc < item_desc_box[3] - 10:
            continue
        filtered.append(p)
    if filtered:
        candidates = filtered

    def rank(p):
        xc, yc = _box_center(p)
        return 0.55 * p[4] + 0.25 * (yc / img_h) + 0.20 * (xc / img_w)

    return max(candidates, key=rank)


def postprocess_predictions(preds):
    """Per-class spatial heuristics → at most one box per class."""
    invoice_number = _pick_best_invoice_number(preds)
    invoice_date = _pick_best_invoice_date(preds)
    seller = _pick_best_seller(preds)
    item_desc = _pick_best_item_description(preds)
    total_amount = _pick_best_total_amount(preds, item_desc_box=item_desc)

    final = []
    for p in [seller, invoice_number, invoice_date, item_desc, total_amount]:
        if p is not None:
            final.append(p)
    return final


# ═══════════════════════════════════════════════════════════════════════════
# 4. Box refinement — expand boxes with class-specific padding
# ═══════════════════════════════════════════════════════════════════════════

def _expand_box(p, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0,
                img_w=IMG_SIZE, img_h=IMG_SIZE):
    x1, y1, x2, y2, score, cls_id = p
    return [
        max(0, x1 - pad_left),
        max(0, y1 - pad_top),
        min(img_w, x2 + pad_right),
        min(img_h, y2 + pad_bottom),
        score,
        cls_id,
    ]


def refine_final_boxes(final_preds):
    refined = []
    for p in final_preds:
        cls_id = p[5]
        if cls_id == TOTAL_AMOUNT_ID:
            p = _expand_box(p, 18, 2, 22, 4)
        elif cls_id == ITEM_DESCRIPTION_ID:
            p = _expand_box(p, 25, 6, 20, 10)
        elif cls_id == SELLER_ID:
            p = _expand_box(p, 6, 4, 8, 6)
        refined.append(p)
    return refined


# ═══════════════════════════════════════════════════════════════════════════
# 5. OCR — crop from tensor, preprocess, tesseract
# ═══════════════════════════════════════════════════════════════════════════

def _crop_box_from_tensor(image_tensor: torch.Tensor, box: list) -> torch.Tensor:
    """Crop [C,H,W] tensor by [x1,y1,x2,y2,score,cls_id]."""
    x1 = int(max(0, round(box[0])))
    y1 = int(max(0, round(box[1])))
    x2 = int(round(box[2]))
    y2 = int(round(box[3]))
    return image_tensor[:, y1:y2, x1:x2]


def _tensor_to_ocr_image(crop_tensor: torch.Tensor) -> np.ndarray:
    """Denormalize (mean=0.5, std=0.5) → uint8."""
    img = crop_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _preprocess_crop_for_ocr(crop_tensor: torch.Tensor, upscale: int = 3):
    """Grayscale → upscale → two binarisation thresholds."""
    img = _tensor_to_ocr_image(crop_tensor)
    if img.ndim == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    pil_img = Image.fromarray(gray)
    pil_img = pil_img.resize((w * upscale, h * upscale), Image.LANCZOS)
    gray = np.array(pil_img)

    bw1 = np.where(gray > 170, 255, 0).astype(np.uint8)
    bw2 = np.where(gray > 200, 255, 0).astype(np.uint8)
    return gray, bw1, bw2


def _run_ocr_on_crop(crop_tensor: torch.Tensor, psm: int = 6,
                     whitelist: str | None = None) -> str:
    import pytesseract

    if crop_tensor.numel() == 0:
        return ""

    gray, bw1, bw2 = _preprocess_crop_for_ocr(crop_tensor, upscale=3)

    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'

    variants = []
    for img_variant in [gray, bw1, bw2]:
        try:
            text = pytesseract.image_to_string(img_variant, config=config).strip()
            variants.append(text)
        except Exception:
            variants.append("")

    # Pick the most content-rich variant
    variants.sort(key=lambda x: len(x.strip()), reverse=True)
    return variants[0] if variants else ""


# ═══════════════════════════════════════════════════════════════════════════
# 6. Text cleaning — exact port from notebook
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_spaces(text: str) -> str:
    text = text.replace("\r", "\n").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_invoice_number(text: str) -> str | None:
    text = _normalize_spaces(text)
    patterns = [
        r"(?i)invoice\s*(?:no|number)?\s*[:#\-]*\s*([A-Za-z0-9][A-Za-z0-9/#\-]{3,})",
        r"(?i)invoiceno\s*([A-Za-z0-9][A-Za-z0-9/#\-]{3,})",
        r"(?i)involce\s*[:#\-]*\s*([A-Za-z0-9][A-Za-z0-9/#\-]{3,})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            candidate = m.group(1).strip(" .,:;#-_/")
            candidate = re.sub(r"^[A-Za-z](\d{4,})$", r"\1", candidate)
            return candidate if candidate else None

    cleaned = re.sub(r"(?i)invoice\s*(?:no|number)?\s*[:#\-]*", " ", text)
    cleaned = re.sub(r"(?i)invoiceno", " ", cleaned)
    cleaned = re.sub(r"(?i)involce", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;#-_/")

    candidates = re.findall(r"[A-Za-z0-9/#\-]{4,}", cleaned)
    candidates = [c for c in candidates if any(ch.isdigit() for ch in c)]
    if not candidates:
        return None

    def rank_token(s):
        digit_count = sum(ch.isdigit() for ch in s)
        alpha_count = sum(ch.isalpha() for ch in s)
        return (digit_count, -alpha_count, len(s))

    best = sorted(candidates, key=rank_token, reverse=True)[0]
    best = re.sub(r"^[A-Za-z](\d{4,})$", r"\1", best)
    return best if best else None


def clean_invoice_date(text: str) -> str | None:
    text = _normalize_spaces(text)
    text = re.sub(r"(?i)(invoice\s*date|date\s*of\s*issue|issue\s*date|date)\s*[:\-]?\s*", "", text)
    text = text.strip(" .,:;")
    text = text.replace(" ,", ",")
    text = re.sub(r"\b([A-Za-z]{3,15})(\d{1,2}),(\d{4})\b", r"\1 \2, \3", text)
    text = re.sub(r"\b([A-Za-z]{3,15})\s+(\d)\s+(\d),\s*(\d{4})\b", r"\1 \2\3, \4", text)
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"^[A-Za-z](?=[A-Z][a-z]{2,})", "", text)

    patterns = [
        r"\b\d{2}[./-]\d{2}[./-]\d{4}\b",
        r"\b\d{4}[./-]\d{2}[./-]\d{2}\b",
        r"\b[A-Za-z]{3,15}\s+\d{1,2},\s+\d{4}\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(0)
    return text if text else None


def clean_total_amount(text: str) -> str | None:
    lines = [l.strip() for l in text.replace("\r", "\n").split("\n") if l.strip()]
    for line in reversed(lines):
        matches = re.findall(r"\d[\d\s]*[.,]\d{2}", line)
        if matches:
            return matches[-1].replace(" ", "")
    text_flat = _normalize_spaces(text)
    matches = re.findall(r"\d[\d\s]*[.,]\d{2}", text_flat)
    if matches:
        return matches[-1].replace(" ", "")
    matches2 = re.findall(r"\d+", text_flat)
    if matches2:
        return matches2[-1]
    return text_flat if text_flat else None


def clean_seller(text: str) -> str | None:
    lines = [l.strip() for l in text.replace("\r", "\n").split("\n") if l.strip()]
    cleaned_lines = []
    for line in lines:
        low = line.lower()
        if any(kw in low for kw in ["tax id", "iban", "items", "client"]):
            break
        # Strip OCR variants: "BIL tO", "BILL T0", "BILLTO", "Bill to", etc.
        line = re.sub(r"(?i)bil+\s*t[o0]\s*[:\-]?\s*", "", line).strip()
        line = re.sub(r"(?i)^(seller|bill\s*from|from|vendor|supplier|поставщик|продавец)\s*[:\-]?\s*", "", line).strip()
        if line:
            cleaned_lines.append(line)
    result = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
    return result if len(result) >= 3 else None


def clean_item_description(text: str) -> str | None:
    lines = [l.strip() for l in text.replace("\r", "\n").split("\n") if l.strip()]
    filtered = []
    for line in lines:
        low = line.lower()
        if any(k in low for k in [
            "description", "qty", "um", "net price", "net worth",
            "vat", "gross", "worth", "price", "total",
        ]):
            continue
        if len(line) < 4:
            continue
        letters = sum(ch.isalpha() for ch in line)
        digits = sum(ch.isdigit() for ch in line)
        if letters < 2:
            continue
        if digits > letters * 1.5:
            continue
        line = re.sub(r"^\s*[0-9Oo]+[.)]?\s*", "", line)
        filtered.append(line)
    deduped = []
    for line in filtered:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    result = "\n".join(deduped).strip()
    return result if len(result) >= 3 else None


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def process_document(image: Image.Image) -> dict:
    """
    Run full pipeline on a PIL image.

    Returns dict with keys:
        seller, invoice_number, invoice_date, total_amount,
        item_description, confidence
    """
    logger.info("Processing document %dx%d px", *image.size)

    model = load_model()
    device = torch.device(settings.DEVICE)

    # 1. Preprocess (512×512, mean=0.5/std=0.5)
    image_tensor = preprocess_image(image)  # [3, 512, 512]
    batch = image_tensor.unsqueeze(0).to(device)

    # 2. Model forward → list of dicts
    with torch.no_grad():
        outputs = model(batch)

    # 3. Decode
    raw_preds = decode_fcos_predictions(
        outputs,
        img_size=IMG_SIZE,
        score_threshold=0.2,
        iou_threshold=0.5,
        top_k=30,
    )
    logger.info("Raw detections: %d", len(raw_preds))
    for p in raw_preds[:10]:
        logger.info(
            "  %s  score=%.3f  box=[%.0f,%.0f,%.0f,%.0f]",
            ID2LABEL[p[5]], p[4], *p[:4],
        )

    # 4. Per-class spatial postprocessing
    final_preds = postprocess_predictions(raw_preds)
    logger.info("After spatial filtering: %d fields", len(final_preds))

    # 5. Expand boxes
    refined_preds = refine_final_boxes(final_preds)

    # 6. OCR + clean per field
    fields: dict = {
        "seller": None,
        "invoice_number": None,
        "invoice_date": None,
        "total_amount": None,
        "item_description": None,
    }
    confidence: dict = {}

    for p in refined_preds:
        x1, y1, x2, y2, score, cls_id = p
        name = ID2LABEL[cls_id]
        confidence[name] = round(score, 4)

        crop = _crop_box_from_tensor(image_tensor, p)
        if crop.numel() == 0:
            raw_text = ""
        elif cls_id == INVOICE_NUMBER_ID:
            raw_text = _run_ocr_on_crop(
                crop, psm=7,
                whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/#",
            )
        elif cls_id == INVOICE_DATE_ID:
            raw_text = _run_ocr_on_crop(
                crop, psm=7,
                whitelist="0123456789/.-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,",
            )
        elif cls_id == TOTAL_AMOUNT_ID:
            raw_text = _run_ocr_on_crop(
                crop, psm=7,
                whitelist="0123456789.,$€£ ",
            )
        elif cls_id == SELLER_ID:
            raw_text = _run_ocr_on_crop(crop, psm=6)
        elif cls_id == ITEM_DESCRIPTION_ID:
            raw_text = _run_ocr_on_crop(crop, psm=6)
        else:
            raw_text = _run_ocr_on_crop(crop, psm=6)

        logger.info("OCR [%s] raw: %r", name, raw_text[:120])

        # Clean
        if cls_id == SELLER_ID:
            fields["seller"] = clean_seller(raw_text)
        elif cls_id == INVOICE_NUMBER_ID:
            fields["invoice_number"] = clean_invoice_number(raw_text)
        elif cls_id == INVOICE_DATE_ID:
            fields["invoice_date"] = clean_invoice_date(raw_text)
        elif cls_id == TOTAL_AMOUNT_ID:
            fields["total_amount"] = clean_total_amount(raw_text)
        elif cls_id == ITEM_DESCRIPTION_ID:
            fields["item_description"] = clean_item_description(raw_text)

    # Fill missing confidence
    for name in ID2LABEL.values():
        if name not in confidence:
            confidence[name] = 0.0

    fields["confidence"] = confidence
    logger.info("Final fields: %s", {k: v for k, v in fields.items() if k != "confidence" and v})
    return fields
