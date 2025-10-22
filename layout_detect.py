# layout_detect.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import re

from first_step_loader import PageRep

# NEW: cv2 / numpy / fitz for visual figure detection
import io
import numpy as np
import cv2
import fitz  # PyMuPDF


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
    Render the page, detect large thin rectangular frames (figure borders),
    and return their bboxes in PDF user-space coordinates.
    """
    # --- render page to PNG ---
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    doc.close()

    # PNG -> numpy (BGR)
    import numpy as np, cv2
    data = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    # --- edge map that likes thin black borders ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # amplify contrast around dark lines
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)

    # --- contours ---
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cand_px: List[Tuple[int, int, int, int]] = []
    page_area_px = W * H

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # size gates (looser than before)
        if area < 0.015 * page_area_px or area > 0.70 * page_area_px:
            continue
        if w < 120 or h < 120:
            continue

        # rectangularity via polygon approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        # figures usually not glued to page edges
        if x < 15 or y < 15 or (x + w) > (W - 15) or (y + h) > (H - 15):
            continue

        # aspect sanity
        ar = w / float(h)
        if not (0.55 <= ar <= 2.8):
            continue

        cand_px.append((x, y, x + w, y + h))

    # sort top-to-bottom, left-to-right for stability
    cand_px.sort(key=lambda r: (r[1], r[0]))

    # --- convert to PDF user-space ---
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
                       dpi: int = 150) -> None:
    """Crop each candidate bbox, remove any text rectangles inside, and save."""
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

        # PDF→px scales for this crop
        sx = W / rect.width
        sy = H / rect.height

        # build mask from provided text blocks that overlap this figure
        mask = np.zeros((H, W), dtype=np.uint8)
        for tb in text_blocks_for_masks:
            tbbox = tuple(tb["bbox"])
            if _ioa_block_in_fig(tbbox, bbox) < 0.5:
                continue
            bx0, by0, bx1, by1 = tbbox
            px0 = int(np.clip((bx0 - rect.x0) * sx, 0, W))
            py0 = int(np.clip((by0 - rect.y0) * sy, 0, H))
            px1 = int(np.clip((bx1 - rect.x0) * sx, 0, W))
            py1 = int(np.clip((by1 - rect.y0) * sy, 0, H))
            if px1 - px0 > 2 and py1 - py0 > 2:
                mask[py0:py1, px0:px1] = 255

        if np.any(mask):
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 1)
            crop = cv2.inpaint(crop, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

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


def detect_layout(
    rep: PageRep,
    *,
    pdf_path: Optional[str] = None,
    page_number: Optional[int] = None,
) -> Dict[str, Any]:
    """Rule-based layout labeling of text blocks + figure candidates (CV fallback)."""
    labeled: List[LabeledBlock] = []

    for tb in rep.text_blocks:
        t = tb.text.strip()
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
        labeled.append(LabeledBlock(type=label, bbox=tb.bbox, text=t))

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
                if ry1 > y0 + 12:        # tolerance for rasterization wobble
                    continue
                # need some horizontal agreement with (expanded) caption band
                if _overlap_ratio_to(r, exp_left, exp_right) < 0.30:
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
        _save_figure_crops(
            pdf_path, page_number, figure_candidates,
            text_blocks_for_masks=removed_inside,  # use the ones we just removed
            out_dir="figures", dpi=150
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
