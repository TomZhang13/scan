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
    type: str  # "caption" | "legend_item" | "status" | "section_header" | "body_text"
    bbox: Tuple[float, float, float, float]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "bbox": self.bbox, "text": self.text}


CAPTION_RE = re.compile(r"^Figure\s+\d+-\d+\b", re.IGNORECASE)
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


# ---------- NEW: detect rectangular figure frames via OpenCV ----------
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

    # ADAPTIVE GATES (was: area >= 1.5%, w/h >= 120px, aspect 0.55..2.8)
    min_area = 0.005 * page_area_px          # 0.5% of page (down from 1.5%)
    max_area = 0.75 * page_area_px           # keep generous upper bound
    min_dim_px = max(64, int(0.018 * min(W, H)))  # ~1.8% of min page dim or 64px
    min_ar, max_ar = 0.40, 4.50              # widen allowed aspect range

    cand_px: List[Tuple[int, int, int, int]] = []
    pad_from_edge = max(8, int(0.006 * min(W, H)))  # was fixed 15px

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


def _overlap_ratio_to(rect: Tuple[float,float,float,float], x0: float, x1: float) -> float:
    """Return fraction of rect's width that overlaps [x0,x1]."""
    rx0, _, rx1, _ = rect
    inter = max(0.0, min(rx1, x1) - max(rx0, x0))
    rw = max(1e-6, rx1 - rx0)
    return inter / rw

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

_FID_RE = re.compile(r"(\d+-\d+)")

def _extract_fid(caption_text: str | None, fallback: str) -> str:
    if caption_text:
        m = _FID_RE.search(caption_text)
        if m:
            return m.group(1)
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
        "legend_item": (255, 149, 0, 255),
        "status": (175, 82, 222, 255),
        "section_header": (52, 199, 89, 255),
        "body_text": (142, 142, 147, 180),
        "figure_candidate": (255, 59, 48, 255),
        "figure_candidate_cv": (255, 59, 48, 255),
        "figure_candidate_cv_matched": (255, 59, 48, 255),
        # include if you added the synth fallback:
        "figure_candidate_synth": (255, 204, 0, 255),
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

    # Save PNG overlay
    base.save(png_path)

def _ocr_text_blocks(pdf_path: str,
                     page_number: int,
                     page_w: float,
                     page_h: float,
                     dpi: int = 400) -> List[Dict[str, Any]]:
    """
    OCR with stronger junk suppression:
      - require higher word confidence
      - ignore very small/skinny words (edge/holes speckles)
      - per-segment density/area/height/median-conf filters
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

    # --- preprocessing: grayscale -> CLAHE -> light sharpen (same as before) ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, ksize=(0, 0), sigmaX=0.8)
    prep = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)

    # ---- tunables (adjust via env if needed) ----
    # ---- tunables (balanced defaults; can override via env) ----
    import os
    MIN_WORD_CONF = int(os.getenv("MIN_WORD_CONF", "66"))   # was 72
    MIN_WORD_W_PX = int(os.getenv("MIN_WORD_W_PX", "5"))    # was 6
    MIN_WORD_H_PX = int(os.getenv("MIN_WORD_H_PX", "7"))    # was 8
    MAX_WORD_H_FRAC = float(os.getenv("MAX_WORD_H_FRAC", "0.15"))  # was 0.10

    MIN_SEG_H_PX = int(os.getenv("MIN_SEG_H_PX", "8"))      # was 10
    MIN_SEG_AREA_FRAC = float(os.getenv("MIN_SEG_AREA_FRAC", "1.0e-5"))  # was 2.0e-5
    MIN_SEG_CHARS = int(os.getenv("MIN_SEG_CHARS", "3"))    # unchanged
    MIN_CHARS_PER_PX = float(os.getenv("MIN_CHARS_PER_PX", "0.006"))      # was 0.008
    MIN_SEG_MEDIAN_CONF = int(os.getenv("MIN_SEG_MEDIAN_CONF", "64"))     # was 70


    # helper: run one tess pass and return segments
    def ocr_pass(psm: int, min_conf: int) -> List[Dict[str, Any]]:
        data = pytesseract.image_to_data(
            prep,
            output_type=pytesseract.Output.DICT,
            config=f"--oem 3 --psm {psm} -l eng"
        )
        n = len(data["text"])

        # group words by (block, paragraph, line)
        from collections import defaultdict
        lines = defaultdict(list)
        for i in range(n):
            txt = (data["text"][i] or "").strip()

            # parse conf; tesseract can emit -1 or strings
            conf_raw = data["conf"][i]
            try:
                conf = int(float(conf_raw))
            except (ValueError, TypeError):
                conf = -1

            if not txt:
                continue
            # reject punctuation-only fragments early
            if re.fullmatch(r"[\W_]+", txt):
                continue
            # hard confidence gate for words (previously we allowed low-conf w/ alnum)
            if conf < min_conf:
                continue

            w = int(data["width"][i]); h = int(data["height"][i])
            # kill tiny specks and ultra-tall blobs (edges/holes)
            if w < MIN_WORD_W_PX or h < MIN_WORD_H_PX or (h > MAX_WORD_H_FRAC * H):
                continue

            l = int(data["left"][i]); t = int(data["top"][i])
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines[key].append({"text": txt, "x0": l, "y0": t, "x1": l + w, "y1": t + h, "conf": conf})

        # pixel → PDF scaling
        sx = page_w / float(W)
        sy = page_h / float(H)

        segs: List[Dict[str, Any]] = []
        for words in lines.values():
            if not words:
                continue
            words.sort(key=lambda w: w["x0"])

            widths = [w["x1"] - w["x0"] for w in words]
            med_w = float(np.median(widths)) if widths else 0.0
            GAP_MIN_PX = 24.0
            GAP_FACTOR = 0.8
            gap_thresh = max(GAP_MIN_PX, GAP_FACTOR * med_w)

            # split by large gaps
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

            # emit filtered segments
            for seg in segments:
                text = " ".join(w["text"] for w in seg).strip()
                if not text:
                    continue

                x0 = min(w["x0"] for w in seg); y0 = min(w["y0"] for w in seg)
                x1 = max(w["x1"] for w in seg); y1 = max(w["y1"] for w in seg)
                bw = max(1, x1 - x0); bh = max(1, y1 - y0)
                area_frac = (bw * bh) / float(W * H)

                # segment-level rejections: too small, too short, too sparse, too few chars
                if bh < MIN_SEG_H_PX:
                    continue
                if area_frac < MIN_SEG_AREA_FRAC:
                    continue
                if len(text) < MIN_SEG_CHARS and not re.search(r"[A-Za-z]{2,}|[0-9]{2,}", text):
                    continue
                if (len(text) / float(bw)) < MIN_CHARS_PER_PX:
                    continue

                # require reasonable median confidence across words in the segment
                median_conf = float(np.median([w["conf"] for w in seg]))
                if median_conf < MIN_SEG_MEDIAN_CONF:
                    continue

                segs.append({
                    "text": text,
                    "bbox": (x0 * sx, y0 * sy, x1 * sx, y1 * sy),
                })
        return segs

    # run two passes; slightly different base thresholds
    # run two passes with slightly softer gates
    segs_6 = ocr_pass(psm=6, min_conf=max(MIN_WORD_CONF, 66))
    segs_4 = ocr_pass(psm=4, min_conf=max(MIN_WORD_CONF - 6, 60))

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

    # --- NEW: CV-based detection if none were found and we have the PDF path ---
    if not figure_candidates and pdf_path is not None and page_number is not None:
        rects = _detect_fig_rects_via_cv(pdf_path, page_number, rep.width, rep.height, dpi=150)

        captions = [b for b in labeled if b.type == "caption"]
        used: List[Tuple[float,float,float,float]] = []

        for cap in captions:
            x0, y0, x1, y1 = cap.bbox

            # expand caption x-range generously to the right (figures are wider)
            exp_left  = max(0.0, x0 - 40.0)
            exp_right = min(rep.width, x1 + 220.0)

            cand = []
            for r in rects:
                rx0, ry0, rx1, ry1 = r
                # must be roughly ABOVE caption (allow a little tolerance)
                # allow a bit more vertical slack between figure bottom and caption top
                if ry1 > y0 + 25:
                    continue
                # allow narrower figures to still match their caption band
                if _overlap_ratio_to(r, exp_left, exp_right) < 0.20:
                    continue

                dy = y0 - ry1             # distance from figure bottom to caption top
                cand.append((dy, r))

            if cand:
                cand.sort(key=lambda t: t[0])     # closest above wins
                chosen = cand[0][1]
                # avoid duplicates if already picked
                if all(_iou(chosen, u) < 0.6 for u in used):
                    figure_candidates.append({
                        "type": "figure_candidate_cv_matched",
                        "bbox": chosen,
                        "caption_text": cap.text
                    })
                    used.append(chosen)


    # --- remove text blocks that lie inside any figure (so overlay won't show them) ---
    blocks_all = [b.to_dict() for b in labeled]
    removed_inside: List[Dict[str, Any]] = []
    kept_blocks: List[Dict[str, Any]] = []

    if figure_candidates:
        for b in blocks_all:
            t = b.get("type")
            # only purge general text; captions/legend/status live outside figures
            if t == "body_text":
                if any(_ioa_block_in_fig(tuple(b["bbox"]), tuple(fc["bbox"])) >= 0.5
                       for fc in figure_candidates):
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
        "legend_item": (255, 149, 0, 255),  # orange
        "status": (175, 82, 222, 255),  # purple
        "section_header": (52, 199, 89, 255),  # green
        "body_text": (142, 142, 147, 180),  # gray
        "figure_candidate": (255, 59, 48, 255),  # red
        "figure_candidate_cv": (255, 59, 48, 255),
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
            "legend_items": sum(1 for b in layout["blocks"] if b["type"] == "legend_item"),
            "status": sum(1 for b in layout["blocks"] if b["type"] == "status"),
            "section_headers": sum(1 for b in layout["blocks"] if b["type"] == "section_header"),
            "body_text": sum(1 for b in layout["blocks"] if b["type"] == "body_text"),
        },
        "figure_candidates": len(layout["figure_candidates"]),
    })

    doc.close()
