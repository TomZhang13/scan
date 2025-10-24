# layout_detect.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import re

from first_step_loader import PageRep

# cv2 / numpy / fitz for visual figure detection
import io
import numpy as np
import cv2
import fitz  # PyMuPDF
import pytesseract
# https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\tomz\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# ---- Simple labeled block ----
@dataclass
class LabeledBlock:
    type: str  # "caption" | "table_caption" | "legend_item" | "status" | "section_header" | "body_text"
    bbox: Tuple[float, float, float, float]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "bbox": self.bbox, "text": self.text}


# Allow optional trailing subfigure letter, e.g. Figure 4-27A or Figure 4-27 (A)
CAPTION_RE = re.compile(
    r"""^Figure\s+                 # "Figure" + spaces
        \d+(?:-\d+)+               # 4-27 or 3-2-1 style
        (?:\s*\(?[A-Za-z]\)?)?     # optional A/B/C with optional parentheses/space
        (?=\s|$|[.:;,-])           # next is space/end/punctuation
    """,
    re.IGNORECASE | re.VERBOSE
)
TABLE_CAPTION_RE = re.compile(
    r"""^Table\s+
        \d+(?:-\d+)+               # 1-4 or 3-2-1 style
        (?:\s*\(?[A-Za-z]\)?)?     # optional trailing A/B with/without parentheses
        (?=\s|$|[.:;,-])           # next is space/end/punctuation
    """,
    re.IGNORECASE | re.VERBOSE
)
LEGEND_RE = re.compile(r"^\s*(\d+)[\.\)]\s+")
SID_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+")
STATUS_RE = re.compile(r"\b(Acceptable|Target|Defect|Non[- ]?conforming)\b", re.IGNORECASE)


def _area(b):
    x0, y0, x1, y1 = b
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _horiz_overlap_ratio(a: Tuple[float,float,float,float],
                         b: Tuple[float,float,float,float]) -> float:
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    aw = max(1e-6, ax1 - ax0)
    return inter / aw


def _overlap_area(a: Tuple[float,float,float,float],
                  b: Tuple[float,float,float,float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih

def _ioa_block_in_fig(block: Tuple[float,float,float,float],
                      fig: Tuple[float,float,float,float]) -> float:
    """Intersection over block area."""
    inter = _overlap_area(block, fig)
    bx0, by0, bx1, by1 = block
    ba = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / ba


def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    aa = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    bb = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / max(1e-6, aa + bb - inter)

import os

# Capture the numeric part and an optional trailing letter (with/without parentheses)
_FID_RE = re.compile(r"(\d+(?:-\d+)+)\s*(?:\(?([A-Za-z])\)?)?")

def _extract_fid(caption_text: str | None, fallback: str) -> str:
    if caption_text:
        m = _FID_RE.search(caption_text)
        if m:
            base = m.group(1)
            suf = m.group(2) or ""
            return f"{base}{suf}"
    return fallback

# For tables we use the same id extractor (numeric groups + optional letter).
_TID_RE = _FID_RE
def _extract_tid(caption_text: str | None, fallback: str) -> str:
    if caption_text:
        m = _TID_RE.search(caption_text)
        if m:
            base = m.group(1)
            suf = m.group(2) or ""
            return f"{base}{suf}"
    return fallback


def _save_figure_crops(pdf_path: str,
                       page_number: int,
                       candidates: List[Dict[str, Any]],
                       *,
                       text_blocks_for_masks: List[Dict[str, Any]],
                       out_dir: str = "figures",
                       dpi: int = 150,
                       erase_text: bool = False,
                       max_mask_frac: float = 0.06,    # hard safety cap: 6% of crop area
                       dilate_kernel: int = 3,
                       inpaint_radius: int = 2,
                       debug_masks: bool = False) -> None:
    """
    Crop each candidate bbox and (optionally) inpaint tiny text rectangles inside.
    Inpainting is OFF by default to avoid smearing figures.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    for i, cand in enumerate(candidates):
        bbox = cand.get("bbox")
        if not bbox:
            continue
        rect = fitz.Rect(bbox)

        # render crop
        pix = page.get_pixmap(clip=rect, dpi=dpi)
        img_bytes = pix.tobytes("png")
        arr = np.frombuffer(img_bytes, np.uint8)
        crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        H, W = crop.shape[:2]
        fig_area = float(W * H)

        # PDF→px scale for this crop
        sx = W / rect.width
        sy = H / rect.height

        # Build mask only if erase_text requested
        mask = np.zeros((H, W), dtype=np.uint8)
        if erase_text and text_blocks_for_masks:
            for tb in text_blocks_for_masks:
                tbbox = tuple(tb["bbox"])
                # only consider blocks that are clearly inside this figure
                if _ioa_block_in_fig(tbbox, bbox) < 0.5:
                    continue

                bx0, by0, bx1, by1 = tbbox
                px0 = int(np.clip((bx0 - rect.x0) * sx, 0, W))
                py0 = int(np.clip((by0 - rect.y0) * sy, 0, H))
                px1 = int(np.clip((bx1 - rect.x0) * sx, 0, W))
                py1 = int(np.clip((by1 - rect.y0) * sy, 0, H))
                w = px1 - px0
                h = py1 - py0
                if w <= 2 or h <= 2:
                    continue

                # extra guards: keep only "text-like" small, skinny rectangles
                #   - small area vs. figure
                #   - short-ish height vs. figure
                #   - reasonable aspect ratio
                area_frac = (w * h) / max(1.0, fig_area)
                if area_frac > 0.02:          # skip blocks larger than 2% of the figure
                    continue
                if h > 0.12 * H:              # skip tall boxes
                    continue
                if w < 10:                    # too thin to be meaningful
                    continue

                mask[py0:py1, px0:px1] = 255

            if np.any(mask):
                # small dilate to connect characters
                if dilate_kernel > 1:
                    mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8), 1)

                mask_frac = float(np.count_nonzero(mask)) / fig_area

                # only inpaint if the mask is small enough (prevents visible smears)
                if mask_frac <= max_mask_frac:
                    crop = cv2.inpaint(crop, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
                else:
                    if debug_masks:
                        fid_dbg = _extract_fid(cand.get("caption_text"), f"p{page_number+1}_{i+1}")
                        cv2.imwrite(os.path.join(out_dir, f"mask_too_big_{fid_dbg}.png"), mask)

        # save file
        fid = _extract_fid(cand.get("caption_text"), f"p{page_number+1}_{i+1}")
        out_path = os.path.join(out_dir, f"fig_{fid}.png")
        n = 2
        while os.path.exists(out_path):
            out_path = os.path.join(out_dir, f"fig_{fid}_{n}.png"); n += 1
        cv2.imwrite(out_path, crop)
        cand["image_uri"] = os.path.abspath(out_path)

    doc.close()


def _save_table_crops(pdf_path: str,
                      page_number: int,
                      candidates: List[Dict[str, Any]],
                      *,
                      out_dir: str = "tables",
                      dpi: int = 150) -> None:
    """
    Crop and save each table bbox as PNG. No inpainting—keep grid/text intact.
    Writes absolute path to cand['image_uri'].
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    for i, cand in enumerate(candidates):
        bbox = cand.get("bbox")
        if not bbox:
            continue
        rect = fitz.Rect(bbox)
        pix = page.get_pixmap(clip=rect, dpi=dpi)
        img_bytes = pix.tobytes("png")
        # filename by tid if present in caption
        tid = _extract_tid(cand.get("caption_text"), f"p{page_number+1}_{i+1}")
        out_path = os.path.join(out_dir, f"table_{tid}.png")
        n = 2
        while os.path.exists(out_path):
            out_path = os.path.join(out_dir, f"table_{tid}_{n}.png"); n += 1
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        cand["image_uri"] = os.path.abspath(out_path)

    doc.close()


import os, io, json
from PIL import Image, ImageDraw

def _save_layout_assets(pdf_path: str,
                        page_number: int,
                        layout: Dict[str, Any],
                        out_dir: str = "layouts",
                        dpi: int = 150) -> None:
    import os, io, json
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)

    # --- name files by the PDF's basename ---
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(out_dir, f"{pdf_stem}_p{page_number:03d}.json")
    png_path  = os.path.join(out_dir, f"{pdf_stem}_p{page_number:03d}.png")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)

    # Render page
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    page_w, page_h = page.rect.width, page.rect.height
    pix = page.get_pixmap(dpi=dpi)
    doc.close()

    # PIL image
    img_bytes = pix.tobytes("png")
    base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(base)

    # PDF user-space -> pixel scale
    sx = base.width / page_w
    sy = base.height / page_h

    def to_px(bbox):
        x0, y0, x1, y1 = bbox
        return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

    COLORS = {
        "caption": (0, 122, 255, 255),
        "table_caption": (255, 214, 10, 255),  # yellow for table captions
        "legend_item": (255, 149, 0, 255),
        "status": (175, 82, 222, 255),
        "section_header": (52, 199, 89, 255),
        "body_text": (142, 142, 147, 180),
        "figure_candidate": (255, 59, 48, 255),
        "figure_candidate_cv": (255, 59, 48, 255),
        "figure_candidate_cv_matched": (255, 59, 48, 255),
        # include if you added the synth fallback:
        "figure_candidate_synth": (255, 204, 0, 255),
        "table_candidate_cv": (255, 214, 10, 255),           # yellow
        "table_candidate_cv_matched": (255, 214, 10, 255),   # yellow
    }

    # Draw text blocks
    for b in layout["blocks"]:
        box = to_px(b["bbox"])
        color = COLORS.get(b["type"], (142, 142, 147, 180))
        draw.rectangle(box, outline=color, width=3)
        x0, y0, *_ = box
        draw.text((x0 + 3, y0 + 3), b["type"], fill=color)

    # Draw figure candidates
    for fc in layout["figure_candidates"]:
        box = to_px(fc["bbox"])
        color = COLORS.get(fc.get("type"), (255, 59, 48, 255))
        draw.rectangle(box, outline=color, width=4)
        x0, y0, *_ = box
        draw.text((x0 + 3, y0 + 3), fc.get("type", "figure_candidate"), fill=color)
 
    # Draw table candidates
    for tc in layout.get("table_candidates", []):
        box = to_px(tc["bbox"])
        color = COLORS.get(tc.get("type"), (255, 214, 10, 255))
        draw.rectangle(box, outline=color, width=4)
        x0, y0, *_ = box
        tag = tc.get("type", "table_candidate")
        draw.text((x0 + 3, y0 + 3), tag, fill=color)

    # Save PNG overlay
    base.save(png_path)


def _overlap_ratio_to(rect: Tuple[float,float,float,float], x0: float, x1: float) -> float:
    """Return fraction of rect's width that overlaps [x0,x1]."""
    rx0, _, rx1, _ = rect
    inter = max(0.0, min(rx1, x1) - max(rx0, x0))
    rw = max(1e-6, rx1 - rx0)
    return inter / rw


# ---------- detect rectangular figure frames via OpenCV ----------
def _detect_fig_rects_via_cv(
    pdf_path: str,
    page_number: int,
    page_w: float,
    page_h: float,
    dpi: int = 150,
) -> List[Tuple[float, float, float, float]]:
    """
    Render the page, detect rectangular frames, and return bboxes in PDF space.
    Adaptively gates min size/aspect so short figures aren't dropped.
    """
    # --- render page to PNG ---
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    doc.close()

    # PNG -> numpy (BGR)
    data = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    page_area_px = W * H

    # --- edges that favor thin borders ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)

    # --- contours ---
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ADAPTIVE GATES (looser defaults; override via env if needed)
    import os
    min_area_frac = float(os.getenv("FIG_MIN_AREA_FRAC", "0.003"))  # was 0.005
    max_area_frac = float(os.getenv("FIG_MAX_AREA_FRAC", "0.85"))   # was 0.75
    min_dim_frac  = float(os.getenv("FIG_MIN_DIM_FRAC",  "0.014"))  # was ~0.018
    min_ar        = float(os.getenv("FIG_MIN_AR",        "0.25"))   # was 0.40
    max_ar        = float(os.getenv("FIG_MAX_AR",        "6.50"))   # was 4.50
    pad_frac      = float(os.getenv("FIG_PAD_FRAC",      "0.004"))  # was 0.006

    min_area = min_area_frac * page_area_px
    max_area = max_area_frac * page_area_px
    min_dim_px = max(48, int(min_dim_frac * min(W, H)))             # was max(64, ...)
    min_ar, max_ar = 0.40, 4.50              # widen allowed aspect range

    cand_px: List[Tuple[int, int, int, int]] = []
    pad_from_edge = max(5, int(pad_frac * min(W, H)))                # was max(8, ...)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < min_area or area > max_area:
            continue
        if w < min_dim_px or h < min_dim_px:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        # avoid page borders
        if x < pad_from_edge or y < pad_from_edge or \
           (x + w) > (W - pad_from_edge) or (y + h) > (H - pad_from_edge):
            continue

        ar = w / float(h)
        if not (min_ar <= ar <= max_ar):
            continue

        cand_px.append((x, y, x + w, y + h))

    cand_px.sort(key=lambda r: (r[1], r[0]))  # top-to-bottom, left-to-right

    # convert to PDF user-space
    sx = page_w / float(W)
    sy = page_h / float(H)
    rects_pdf = [(x0 * sx, y0 * sy, x1 * sx, y1 * sy) for (x0, y0, x1, y1) in cand_px]
    return rects_pdf
 
 
def _nms_rects(rects_px: List[Tuple[int,int,int,int]],
               scores: List[float],
               iou_thresh: float = 0.6) -> List[int]:
    """Simple NMS over pixel-space rects. Returns kept indices."""
    if not rects_px:
        return []
    boxes = np.array(rects_px, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def _detect_table_rects_via_cv(
    pdf_path: str,
    page_number: int,
    page_w: float,
    page_h: float,
    dpi: int = 300,
) -> List[Dict[str, Any]]:
    """
    Return list of table proposals as dicts:
      { "bbox": (x0,y0,x1,y1) in PDF space, "grid_score": float, "brightness": float }
    Approach: binarize -> extract horizontal/vertical line maps -> union -> contours -> filter -> NMS.
    """
    # --- render page ---
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    doc.close()

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- binarize ---
    # Slight blur to reduce speckle; Otsu threshold (invert so lines=1)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- line maps ---
    # Kernel sizes scale with page size
    kx = max(15, W // 60)  # horizontal line length
    ky = max(15, H // 60)  # vertical line length
    horiz = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)), iterations=1)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)), iterations=1)
    vert = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)), iterations=1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)), iterations=1)

    # union of lines; a light dilate to close tiny gaps
    lines = cv2.bitwise_or(horiz, vert)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)

    # edges for border score (thin black frames)
    edges = cv2.Canny(gray, 60, 160)

    # --- find rect proposals on line-union ---
    cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area_px = float(W * H)

    import os
    MIN_AREA_FRAC = float(os.getenv("TABLE_MIN_AREA_FRAC", "0.006"))   # tables can be sizable
    MAX_AREA_FRAC = float(os.getenv("TABLE_MAX_AREA_FRAC", "0.85"))
    MIN_DIM_FRAC  = float(os.getenv("TABLE_MIN_DIM_FRAC",  "0.02"))
    MIN_AR        = float(os.getenv("TABLE_MIN_AR",        "0.35"))
    MAX_AR        = float(os.getenv("TABLE_MAX_AR",        "8.50"))
    PAD_FRAC      = float(os.getenv("TABLE_PAD_FRAC",      "0.004"))
    GRID_THRESH   = float(os.getenv("TABLE_GRID_THRESH",   "0.0045"))  # keep if >=
    BRIGHT_MIN    = float(os.getenv("TABLE_BRIGHT_MIN",    "160"))     # mean gray inside >=
    NMS_IOU       = float(os.getenv("TABLE_NMS_IOU",       "0.60"))

    pad_from_edge = max(5, int(PAD_FRAC * min(W, H)))
    min_area = MIN_AREA_FRAC * page_area_px
    max_area = MAX_AREA_FRAC * page_area_px
    min_dim_px = max(32, int(MIN_DIM_FRAC * min(W, H)))

    rects_px: List[Tuple[int,int,int,int]] = []
    scores: List[float] = []
    brights: List[float] = []

    # integral images for fast box sums
    lines_int = cv2.integral(lines // 255)
    gray_int  = cv2.integral(gray)
    edges_int = cv2.integral(edges // 255)

    def sum_region(ii, x0, y0, x1, y1):
        # inclusive-exclusive on integral image
        x0c, y0c, x1c, y1c = x0, y0, x1-1, y1-1
        return int(ii[y1c+1, x1c+1] - ii[y0c, x1c+1] - ii[y1c+1, x0c] + ii[y0c, x0c])

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or area > max_area:
            continue
        if w < min_dim_px or h < min_dim_px:
            continue
        if x < pad_from_edge or y < pad_from_edge or (x + w) > (W - pad_from_edge) or (y + h) > (H - pad_from_edge):
            continue
        ar = w / float(h)
        if not (MIN_AR <= ar <= MAX_AR):
            continue

        # gridness = line pixels density inside bbox
        line_pix = sum_region(lines_int, x, y, x + w, y + h)
        gridness = line_pix / float(area + 1e-6)
        if gridness < GRID_THRESH:
            continue

        # interior brightness (mean gray)
        gray_sum = sum_region(gray_int, x, y, x + w, y + h)
        mean_gray = gray_sum / float(area + 1e-6)
        if mean_gray < BRIGHT_MIN:
            continue

        # border score: edge pixels along a thin perimeter band
        band = 2
        per_edge = (
            sum_region(edges_int, x, y, x + w, y + band) +
            sum_region(edges_int, x, y + h - band, x + w, y + h) +
            sum_region(edges_int, x, y, x + band, y + h) +
            sum_region(edges_int, x + w - band, y, x + w, y + h)
        ) / float(2*w + 2*h + 1e-6)

        # keep; score mixes gridness and border presence
        score = float(gridness * 0.8 + per_edge * 0.2)
        rects_px.append((x, y, x + w, y + h))
        scores.append(score)
        brights.append(mean_gray)

    # NMS
    keep_idx = _nms_rects(rects_px, scores, iou_thresh=NMS_IOU)
    rects_px = [rects_px[i] for i in keep_idx]
    scores   = [scores[i]   for i in keep_idx]
    brights  = [brights[i]  for i in keep_idx]

    # convert to PDF
    sx = page_w / float(W)
    sy = page_h / float(H)
    out = []
    for (x0, y0, x1, y1), sc, br in zip(rects_px, scores, brights):
        out.append({
            "bbox": (x0 * sx, y0 * sy, x1 * sx, y1 * sy),
            "grid_score": float(sc),
            "brightness": float(br),
        })
    return out


def _ocr_text_blocks(pdf_path: str,
                     page_number: int,
                     page_w: float,
                     page_h: float,
                     dpi: int = 400) -> List[Dict[str, Any]]:
    """
    OCR with stronger junk suppression + two fixes:
      (1) Rescue leading tiny/low-conf words by being more lenient per-word and merging
          tiny leading tokens into the main segment before short-seg filtering.
      (2) Trim segment boxes to ink so right-edge whitespace isn't captured.
    Returns [{'text', 'bbox'}] in PDF space.
    """
    if pytesseract is None:
        return []

    # --- render page ---
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)
    doc.close()

    img_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    H, W = img_bgr.shape[:2]

    # --- preprocessing: grayscale -> CLAHE -> light sharpen ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, ksize=(0, 0), sigmaX=0.8)
    prep = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)

    # ---- tunables (can override via env) ----
    import os
    # Make per-word intake a bit more permissive; rely on segment median conf later
    MIN_WORD_CONF = int(os.getenv("MIN_WORD_CONF", "58"))   # was 66
    MIN_WORD_W_PX = int(os.getenv("MIN_WORD_W_PX", "4"))    # was 5
    MIN_WORD_H_PX = int(os.getenv("MIN_WORD_H_PX", "6"))    # was 7
    MAX_WORD_H_FRAC = float(os.getenv("MAX_WORD_H_FRAC", "0.15"))

    MIN_SEG_H_PX = int(os.getenv("MIN_SEG_H_PX", "8"))
    MIN_SEG_AREA_FRAC = float(os.getenv("MIN_SEG_AREA_FRAC", "1.0e-5"))
    MIN_SEG_CHARS = int(os.getenv("MIN_SEG_CHARS", "3"))
    MIN_CHARS_PER_PX = float(os.getenv("MIN_CHARS_PER_PX", "0.006"))
    MIN_SEG_MEDIAN_CONF = int(os.getenv("MIN_SEG_MEDIAN_CONF", "64"))

    # Ink-trim parameters (avoid big empty bands at edges)
    INK_EMPTY_BAND_MIN = int(os.getenv("INK_EMPTY_BAND_MIN", "6"))  # px
    INK_COL_BLACK_RATIO = float(os.getenv("INK_COL_BLACK_RATIO", "0.012"))  # % of rows

    # final padding (asymmetric: keep right small so we don't hit page edge)
    PAD_LEFT = float(os.getenv("PAD_LEFT_PX", "4.0"))
    PAD_RIGHT = float(os.getenv("PAD_RIGHT_PX", "1.5"))
    PAD_Y = float(os.getenv("PAD_Y_PX", "2.0"))

    # helper: run one tess pass and return segments
    def ocr_pass(psm: int, min_conf: int) -> List[Dict[str, Any]]:
        data = pytesseract.image_to_data(
            prep,
            output_type=pytesseract.Output.DICT,
            config=f"--oem 3 --psm {psm} -l eng"
        )
        n = len(data["text"])

        from collections import defaultdict
        lines = defaultdict(list)
        all_word_w: List[int] = []

        # --- collect words (slightly looser than before) ---
        for i in range(n):
            raw_txt = data["text"][i]
            txt = (raw_txt or "").strip()

            conf_raw = data["conf"][i]
            try:
                conf = int(float(conf_raw))
            except (ValueError, TypeError):
                conf = -1

            # keep low-conf punctuation-only OUT (they explode gaps)
            if not txt:
                continue
            if re.fullmatch(r"[\W_]+", txt):
                continue

            # be a bit more permissive per word;
            # segment-level median conf will prune junk later
            if conf < min_conf:
                # allow *very* low-conf words only if they have reasonable size;
                # they'll be absorbed/trimmed by segment median filter
                if conf < (min_conf - 12):
                    continue

            w = int(data["width"][i]); h = int(data["height"][i])
            if w < MIN_WORD_W_PX or h < MIN_WORD_H_PX or (h > MAX_WORD_H_FRAC * H):
                continue

            l = int(data["left"][i]); t = int(data["top"][i])
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines[key].append({"text": txt, "x0": l, "y0": t, "x1": l + w, "y1": t + h, "conf": conf})
            all_word_w.append(w)

        # PDF user-space scaling
        sx = page_w / float(W)
        sy = page_h / float(H)

        segs_px: List[Dict[str, Any]] = []

        # --- group into segments per visual line ---
        for words in lines.values():
            if not words:
                continue
            words.sort(key=lambda w: w["x0"])

            widths = [w["x1"] - w["x0"] for w in words]
            med_w = float(np.median(widths)) if widths else 0.0

            # tolerate curvature and small breaks
            GAP_MIN_PX = 32.0
            GAP_FACTOR = 1.25
            gap_thresh = max(GAP_MIN_PX, GAP_FACTOR * med_w)

            # split by big gaps
            segments: List[List[Dict[str, Any]]] = []
            cur = [words[0]]
            for prev, curr in zip(words, words[1:]):
                gap = curr["x0"] - prev["x1"]
                if gap > gap_thresh:
                    segments.append(cur)
                    cur = [curr]
                else:
                    cur.append(curr)
            segments.append(cur)

            # Heuristic rescue: merge a *tiny* leading token (e.g., "A", "I", "'The")
            # into the next segment before short-segment filtering.
            def _seg_text_len(seg: List[Dict[str, Any]]) -> int:
                return sum(len(w["text"]) for w in seg)

            if len(segments) >= 2 and _seg_text_len(segments[0]) <= 2:
                lead = segments[0]; nxt = segments[1]
                gap0 = nxt[0]["x0"] - lead[-1]["x1"]
                # be generous but bounded
                if gap0 <= (gap_thresh * 1.75):
                    segments[1] = lead + segments[1]
                    segments = segments[1:]

            # merge neighbors on slightly curved lines (existing logic, a bit looser)
            def _merge_segments_curvy(segs: List[List[Dict[str, Any]]],
                                      base_gap: float,
                                      med_w_local: float) -> List[List[Dict[str, Any]]]:
                if not segs:
                    return segs
                MERGE_EXTRA = 0.55 * max(8.0, med_w_local)
                merged: List[List[Dict[str, Any]]] = [segs[0]]
                for nx in segs[1:]:
                    pv = merged[-1]
                    gap_px = nx[0]["x0"] - pv[-1]["x1"]

                    px0 = min(w["x0"] for w in pv); py0 = min(w["y0"] for w in pv)
                    px1 = max(w["x1"] for w in pv); py1 = max(w["y1"] for w in pv)
                    nx0 = min(w["x0"] for w in nx); ny0 = min(w["y0"] for w in nx)
                    nx1 = max(w["x1"] for w in nx); ny1 = max(w["y1"] for w in nx)
                    ph = max(1, py1 - py0); nh = max(1, ny1 - ny0)

                    v_center_diff = abs(((py0 + py1) / 2.0) - ((ny0 + ny1) / 2.0))
                    v_tol = 0.28 * max(ph, nh)
                    overlap_h = min(py1, ny1) - max(py0, ny0)
                    v_overlap = overlap_h / float(max(1, min(ph, nh)))

                    if (gap_px <= base_gap + MERGE_EXTRA) and (v_center_diff <= v_tol or v_overlap >= 0.55):
                        merged[-1] = pv + nx
                    else:
                        merged.append(nx)
                return merged

            segments = _merge_segments_curvy(segments, gap_thresh, med_w)

            # emit filtered segments (px first)
            for seg in segments:
                text = " ".join(w["text"] for w in seg).strip()
                if not text:
                    continue

                x0 = min(w["x0"] for w in seg); y0 = min(w["y0"] for w in seg)
                x1 = max(w["x1"] for w in seg); y1 = max(w["y1"] for w in seg)
                bw = max(1, x1 - x0); bh = max(1, y1 - y0)
                area_frac = (bw * bh) / float(W * H)

                if bh < MIN_SEG_H_PX:
                    continue
                if area_frac < MIN_SEG_AREA_FRAC:
                    continue
                if len(text) < MIN_SEG_CHARS and not re.search(r"[A-Za-z]{2,}|[0-9]{2,}", text):
                    # this catches isolated 'A'/'I' *after* the rescue merge above
                    continue
                if (len(text) / float(bw)) < MIN_CHARS_PER_PX:
                    continue

                median_conf = float(np.median([w["conf"] for w in seg]))
                if median_conf < MIN_SEG_MEDIAN_CONF:
                    continue

                segs_px.append({
                    "text": text,
                    "bbox_px": (x0, y0, x1, y1),
                    "h_px": bh
                })

        # Optional: stitch across tiny Tesseract line breaks across the page
        if segs_px:
            g_med_w = float(np.median(all_word_w)) if all_word_w else 0.0
            BASE_GAP = max(32.0, 1.25 * g_med_w)
            EXTRA = 0.75 * max(8.0, g_med_w)

            segs_px.sort(key=lambda s: ((s["bbox_px"][1] + s["bbox_px"][3]) / 2.0, s["bbox_px"][0]))
            stitched = [segs_px[0]]
            for s in segs_px[1:]:
                prev = stitched[-1]
                px0, py0, px1, py1 = prev["bbox_px"]
                sx0, sy0, sx1, sy1 = s["bbox_px"]

                v_overlap = max(0.0, min(py1, sy1) - max(py0, sy0)) / float(max(1.0, min(py1 - py0, sy1 - sy0)))
                ph = max(1.0, py1 - py0); nh = max(1.0, sy1 - sy0)
                v_cent_diff = abs((py0 + py1) / 2.0 - (sy0 + sy1) / 2.0)
                v_tol = 0.30 * max(ph, nh)

                hgap = max(0.0, sx0 - px1)
                if (hgap <= BASE_GAP + EXTRA) and (v_overlap >= 0.55 or v_cent_diff <= v_tol):
                    new_bbox = (min(px0, sx0), min(py0, sy0), max(px1, sx1), max(py1, sy1))
                    prev["bbox_px"] = new_bbox
                    prev["text"] = (prev["text"] + " " + s["text"]).strip()
                    prev["h_px"] = max(ph, nh)
                else:
                    stitched.append(s)
            segs_px = stitched

        # --- helper: trim bbox to ink to avoid trailing whitespace ---
        def _tighten_bbox_to_ink(img_gray: np.ndarray,
                                 bbox_px: Tuple[float, float, float, float]) -> Tuple[int,int,int,int]:
            x0, y0, x1, y1 = map(int, bbox_px)
            x0 = max(0, x0); y0 = max(0, y0); x1 = min(img_gray.shape[1]-1, x1); y1 = min(img_gray.shape[0]-1, y1)
            if x1 <= x0 + 1 or y1 <= y0 + 1:
                return x0, y0, x1, y1
            roi = img_gray[y0:y1, x0:x1]
            # Otsu binarize then column projection
            _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            col_sum = (bw > 0).sum(axis=0)
            h = bw.shape[0]
            thr = max(1, int(INK_COL_BLACK_RATIO * h))
            nz = np.where(col_sum >= thr)[0]
            if nz.size == 0:
                return x0, y0, x1, y1
            left_margin = int(nz[0])
            right_margin = int((bw.shape[1]-1) - nz[-1])
            if left_margin >= INK_EMPTY_BAND_MIN:
                x0 += left_margin
            if right_margin >= INK_EMPTY_BAND_MIN:
                x1 -= right_margin
            return x0, y0, x1, y1

        # Final small padding after ink-trim (avoid snapping to page edge)
        out: List[Dict[str, Any]] = []
        for s in segs_px:
            x0, y0, x1, y1 = s["bbox_px"]
            # trim to ink (use preprocessed image for robustness)
            x0, y0, x1, y1 = _tighten_bbox_to_ink(prep, (x0, y0, x1, y1))

            # asymmetric padding; avoid reaching W exactly
            x0 = max(0, int(x0 - PAD_LEFT))
            y0 = max(0, int(y0 - PAD_Y))
            x1 = min(W - 1, int(x1 + PAD_RIGHT))
            y1 = min(H - 1, int(y1 + PAD_Y))

            out.append({
                "text": s["text"],
                "bbox": (x0 * sx, y0 * sy, x1 * sx, y1 * sy),
            })
        return out

    # run two passes with slightly softer gates
    segs_6 = ocr_pass(psm=6, min_conf=max(MIN_WORD_CONF, 58))
    segs_4 = ocr_pass(psm=4, min_conf=max(MIN_WORD_CONF - 6, 52))

    # dedupe merged segments (keep longer text or larger area)
    def iou(a, b):
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        aa = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
        bb = max(1e-6, (bx1 - bx0) * (by1 - by0))
        return inter / max(1e-6, aa + bb - inter)

    merged: List[Dict[str, Any]] = []
    candidates = segs_6 + segs_4
    candidates.sort(
        key=lambda s: (len(s["text"]), (s["bbox"][2]-s["bbox"][0])*(s["bbox"][3]-s["bbox"][1])),
        reverse=True
    )
    for seg in candidates:
        if all(iou(seg["bbox"], m["bbox"]) < 0.55 for m in merged):
            merged.append(seg)

    return merged


def detect_layout(
    rep: PageRep,
    *,
    pdf_path: Optional[str] = None,
    page_number: Optional[int] = None,
) -> Dict[str, Any]:
    """OCR-only text labeling + figure candidates (image/CV), with figure crops + overlay saves."""
    labeled: List[LabeledBlock] = []

    # --- OCR-only text extraction ---
    # (Requires pdf_path + page_number. If missing, no text blocks will be produced.)
    if pdf_path is not None and page_number is not None:
        ocr_lines = _ocr_text_blocks(pdf_path, page_number, rep.width, rep.height, dpi=300)
    else:
        ocr_lines = []

    # Label OCR lines
    for ol in ocr_lines:
        t = (ol.get("text") or "").strip()
        if not t:
            continue
        if CAPTION_RE.match(t):
            label = "caption"
        elif TABLE_CAPTION_RE.match(t):
            label = "table_caption"
        elif LEGEND_RE.match(t):
            label = "legend_item"
        elif SID_RE.match(t):
            label = "section_header"
        elif STATUS_RE.search(t):
            label = "status"
        else:
            label = "body_text"
        labeled.append(LabeledBlock(type=label, bbox=tuple(ol["bbox"]), text=t))

    # figure candidates from embedded images (non–full-page)
    page_area = rep.width * rep.height
    figure_candidates = []
    for ib in rep.image_blocks:
        a = _area(ib.bbox)
        if a < 0.70 * page_area:  # ignore giant background image
            figure_candidates.append({"type": "figure_candidate", "bbox": ib.bbox})

    # --- FIGURES: CV-based detection (ALWAYS run and merge), then match to captions ---
    if pdf_path is not None and page_number is not None:
        rects = _detect_fig_rects_via_cv(pdf_path, page_number, rep.width, rep.height, dpi=150)

        captions = [b for b in labeled if b.type == "caption"]
        # seed 'used' with any existing figure candidates (from embedded images)
        used: List[Tuple[float, float, float, float]] = [tuple(fc["bbox"]) for fc in figure_candidates]

        for cap in captions:
            x0, y0, x1, y1 = cap.bbox

            # widen the horizontal band we consider for this caption
            exp_left  = max(0.0, x0 - 60.0)          # was 40.0
            exp_right = min(rep.width, x1 + 320.0)   # was 220.0

            cand = []
            for r in rects:
                # skip if this rect is basically the same as something we already have
                if any(_iou(r, u) >= 0.60 for u in used):
                    continue

                rx0, ry0, rx1, ry1 = r

                # figure must be roughly ABOVE the caption (allow more slack)
                if ry1 > y0 + 60:    # was +25
                    continue

                # relax required horizontal overlap with the caption band
                if _overlap_ratio_to(r, exp_left, exp_right) < 0.12:  # was 0.20
                    continue

                dy = y0 - ry1  # distance from figure bottom to caption top
                cand.append((dy, r))

            if cand:
                cand.sort(key=lambda t: t[0])  # closest above the caption wins
                chosen = cand[0][1]
                figure_candidates.append({
                    "type": "figure_candidate_cv_matched",
                    "bbox": chosen,
                    "caption_text": cap.text
                })
                used.append(chosen)


    # --- TABLES: detect via CV and associate to captions (strict: keep ONLY caption-matched) ---
    table_candidates: List[Dict[str, Any]] = []
    removed_inside_tables: List[Dict[str, Any]] = []
    if pdf_path is not None and page_number is not None:
        # If no "Table ..." caption on page, skip table detection entirely to avoid false positives.
        t_caps = [b for b in labeled if b.type == "table_caption"]
        if t_caps:
            proposals = _detect_table_rects_via_cv(pdf_path, page_number, rep.width, rep.height, dpi=300)
            for p in proposals:
                table_candidates.append({"type": "table_candidate_cv", "bbox": p["bbox"], "grid_score": p["grid_score"]})

            # geometry-first association: table just below its caption, within a max vertical gap
            import os
            MAX_GAP = float(os.getenv("TABLE_MAX_CAPTION_GAP_PT", "160.0"))  # PDF user space points
            used_tbl: List[Tuple[float,float,float,float]] = []
            for cap in t_caps:
                cx0, cy0, cx1, cy1 = cap.bbox
                exp_left  = max(0.0, cx0 - 60.0)
                exp_right = min(rep.width, cx1 + 320.0)
                cand = []
                for tc in table_candidates:
                    if tc.get("type") != "table_candidate_cv":  # skip already matched
                        continue
                    tx0, ty0, tx1, ty1 = tc["bbox"]
                    # table should start below caption (allow tiny slack)
                    if ty0 < cy1 - 20.0:
                        continue
                    dy = ty0 - cy1
                    # enforce a maximum allowed distance below the caption
                    if dy > MAX_GAP:
                        continue
                    # horizontal overlap with expanded caption band
                    if _overlap_ratio_to(tc["bbox"], exp_left, exp_right) < 0.12:
                        continue
                    # not already used (high IoU)
                    if any(_iou(tc["bbox"], u) >= 0.60 for u in used_tbl):
                        continue
                    cand.append((dy, tc))
                if cand:
                    cand.sort(key=lambda t: t[0])
                    chosen = cand[0][1]
                    chosen["type"] = "table_candidate_cv_matched"
                    chosen["caption_text"] = cap.text
                    used_tbl.append(tuple(chosen["bbox"]))

            # Drop ALL unmatched proposals unless explicitly kept for debugging
            KEEP_UNMATCHED = os.getenv("KEEP_UNMATCHED_TABLES", "0").lower() in {"1","true","yes","on"}
            if not KEEP_UNMATCHED:
                table_candidates = [tc for tc in table_candidates if tc.get("type") == "table_candidate_cv_matched"]

            # Save crops only for the kept (matched or explicitly kept) tables
            if table_candidates:
                _save_table_crops(pdf_path, page_number, table_candidates, out_dir="tables", dpi=150)
        else:
            table_candidates = []  # no table captions on page → no tables


    # --- avoid double labeling vs figures: prefer table when grid strong ---
    if figure_candidates and table_candidates:
        import os
        PREFER_GRID = float(os.getenv("TABLE_GRID_PREFER_THRESH", "0.010"))
        kept_figs = []
        for fc in figure_candidates:
            fb = tuple(fc["bbox"])
            drop = False
            for tc in table_candidates:
                tb = tuple(tc["bbox"])
                if _iou(fb, tb) >= 0.60 and tc.get("grid_score", 0.0) >= PREFER_GRID:
                    drop = True
                    break
            if not drop:
                kept_figs.append(fc)
        figure_candidates = kept_figs

    # --- remove blocks inside tables (drop ALL types), and keep prior figure behavior ---
    blocks_all = [b.to_dict() for b in labeled]
    removed_inside: List[Dict[str, Any]] = []
    kept_blocks: List[Dict[str, Any]] = []

    any_regions = bool(figure_candidates) or bool(table_candidates)
    if any_regions:
        for b in blocks_all:
            # 1) If this block sits inside ANY table region, remove it regardless of type
            inside_tbl = any(
                _ioa_block_in_fig(tuple(b["bbox"]), tuple(tc["bbox"])) >= 0.5
                for tc in table_candidates
            )
            if inside_tbl:
                removed_inside.append(b)
                continue

            # 2) Keep the original policy for figures: only drop body_text inside figures
            if b.get("type") == "body_text":
                inside_fig = any(
                    _ioa_block_in_fig(tuple(b["bbox"]), tuple(fc["bbox"])) >= 0.5
                    for fc in figure_candidates
                )
                if inside_fig:
                    removed_inside.append(b)
                    continue

            kept_blocks.append(b)
    else:
        kept_blocks = blocks_all

    # --- Save figure crops (erase text inside figures) ---
    if pdf_path is not None and page_number is not None and figure_candidates:
        # Inpainting is off by default; enable with env ERASE_FIG_TEXT=1 if desired.
        _do_erase = os.getenv("ERASE_FIG_TEXT", "0").lower() in {"1", "true", "yes", "on"}
        _save_figure_crops(
            pdf_path, page_number, figure_candidates,
            text_blocks_for_masks=removed_inside,
            out_dir="figures", dpi=150,
            erase_text=_do_erase
        )

    result = {
        "page_number": rep.page_number,
        "page_size": (rep.width, rep.height),
        "blocks": kept_blocks,                  # (already filtered of text inside figures)
        "figure_candidates": figure_candidates, # may include "image_uri"
        "table_candidates": table_candidates,   # may include "image_uri", "caption_text", "grid_score"
    }

    # --- ALWAYS save layout assets when we have the inputs ---
    if pdf_path is not None and page_number is not None:
        _save_layout_assets(pdf_path, page_number, result, out_dir="layouts", dpi=150)

    return result


# quick manual run
if __name__ == "__main__":
    import os, json, io
    from pprint import pprint
    from PIL import Image, ImageDraw
    from first_step_loader import load_pdf_page

    # ---- config ----
    pdf_path = r"1-7/4.0/IPC_0018_SCH.pdf"
    page_number = 0
    out_dir = "layouts"
    dpi = 150

    os.makedirs(out_dir, exist_ok=True)

    # 1) build layout (now pass pdf_path + page_number for CV fallback)
    rep = load_pdf_page(pdf_path, page_number=page_number)
    layout = detect_layout(rep, pdf_path=pdf_path, page_number=page_number)

    # 2) save JSON
    json_path = os.path.join(out_dir, f"layout_p{page_number:03d}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)

    # 3) render page and draw ALL boxes
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)

    # convert pixmap -> PIL image
    img_bytes = pix.tobytes("png")
    base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(base)

    # PDF user-space -> pixel scale
    sx = base.width / page.rect.width
    sy = base.height / page.rect.height

    def to_px(bbox):
        x0, y0, x1, y1 = bbox
        return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

    # colors
    COLORS = {
        "caption": (0, 122, 255, 255),  # blue
        "table_caption": (255, 214, 10, 255),  # yellow
        "legend_item": (255, 149, 0, 255),  # orange
        "status": (175, 82, 222, 255),  # purple
        "section_header": (52, 199, 89, 255),  # green
        "body_text": (142, 142, 147, 180),  # gray
        "figure_candidate": (255, 59, 48, 255),  # red
        "figure_candidate_cv": (255, 59, 48, 255),
        "table_candidate_cv": (255, 214, 10, 255),
        "table_candidate_cv_matched": (255, 214, 10, 255),
    }

    # draw text blocks
    for b in layout["blocks"]:
        box = to_px(b["bbox"])
        color = COLORS.get(b["type"], (142, 142, 147, 180))
        draw.rectangle(box, outline=color, width=3)
        x0, y0, *_ = box
        draw.text((x0 + 3, y0 + 3), b["type"], fill=color)

    # draw figure candidates (now includes CV detections)
    for fc in layout["figure_candidates"]:
        box = to_px(fc["bbox"])
        color = COLORS.get(fc.get("type"), (255, 59, 48, 255))
        draw.rectangle(box, outline=color, width=4)
        x0, y0, *_ = box
        tag = fc.get("type", "figure_candidate")
        draw.text((x0 + 3, y0 + 3), tag, fill=color)
 
    # draw table candidates
    for tc in layout.get("table_candidates", []):
        box = to_px(tc["bbox"])
        color = COLORS.get(tc.get("type"), (255, 214, 10, 255))
        draw.rectangle(box, outline=color, width=4)
        x0, y0, *_ = box
        tag = tc.get("type", "table_candidate")
        draw.text((x0 + 3, y0 + 3), tag, fill=color)

    # 4) save overlay image
    png_path = os.path.join(out_dir, f"layout_p{page_number:03d}.png")
    base.save(png_path)

    # summary to console
    from pprint import pprint as _p
    _p({
        "saved_json": json_path,
        "saved_png": png_path,
        "counts": {
            "captions": sum(1 for b in layout["blocks"] if b["type"] == "caption"),
            "table_captions": sum(1 for b in layout["blocks"] if b["type"] == "table_caption"),
            "legend_items": sum(1 for b in layout["blocks"] if b["type"] == "legend_item"),
            "status": sum(1 for b in layout["blocks"] if b["type"] == "status"),
            "section_headers": sum(1 for b in layout["blocks"] if b["type"] == "section_header"),
            "body_text": sum(1 for b in layout["blocks"] if b["type"] == "body_text"),
        },
        "figure_candidates": len(layout["figure_candidates"]),
        "table_candidates": len(layout.get("table_candidates", [])),
    })

    doc.close()
