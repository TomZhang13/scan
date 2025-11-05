from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

# --- stdlib ---
import os, json, re, math

# PDF deps
import fitz  # PyMuPDF

# Image fallback (optional)
from PIL import Image


# ==========================
# Core lightweight structures
# ==========================
@dataclass
class TextBlock:
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF user space


@dataclass
class ImageBlock:
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class PageRep:
    source_type: str               # "pdf" | "image"
    page_number: int
    width: float
    height: float
    text_blocks: List[TextBlock]
    image_blocks: List[ImageBlock]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "text_blocks": [asdict(tb) for tb in self.text_blocks],
            "image_blocks": [asdict(ib) for ib in self.image_blocks],
        }


# ==================
# Minimal page loader
# ==================
def load_pdf_page(pdf_path: str, page_number: int = 0) -> PageRep:
    """Minimal PDF loader: page size, text blocks, and image bboxes."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    # Page geometry (PDF user space, origin bottom-left)
    width, height = page.rect.width, page.rect.height

    # Simple text blocks (block extractor keeps lines together)
    text_blocks: List[TextBlock] = []
    for x0, y0, x1, y1, text, block_no, block_type in page.get_text("blocks"):
        # block_type: 0=text, 1=image, 2=vector, etc.
        if block_type == 0 and (text or "").strip():
            text_blocks.append(TextBlock(text=text.strip(), bbox=(x0, y0, x1, y1)))

    # Image rectangles via xrefs
    image_blocks: List[ImageBlock] = []
    for img in page.get_images(full=True):
        xref = img[0]
        # One image xref can appear multiple times on a page; get all rects
        for rect in page.get_image_rects(xref):
            image_blocks.append(ImageBlock(bbox=(rect.x0, rect.y0, rect.x1, rect.y1)))

    doc.close()
    return PageRep(
        source_type="pdf",
        page_number=page_number,
        width=width,
        height=height,
        text_blocks=text_blocks,
        image_blocks=image_blocks,
    )


def load_image_page(image_path: str) -> PageRep:
    """
    Minimal image fallback: record pixel geometry and a single full-page block.
    Later steps (layout/OCR) will segment this.
    """
    im = Image.open(image_path)
    w, h = im.size  # pixel space
    return PageRep(
        source_type="image",
        page_number=0,
        width=float(w),
        height=float(h),
        text_blocks=[],          # fill after OCR/layout
        image_blocks=[ImageBlock(bbox=(0.0, 0.0, float(w), float(h)))],
    )


# =====================
# JSON Assembly Helpers
# =====================
_SID_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.*)$")
_FID_RE = re.compile(r"Figure\s+(\d+(?:-\d+)+)(?:\s*[A-Za-z]\)?\)?)?", re.IGNORECASE)
_TID_RE = re.compile(r"Table\s+(\d+(?:-\d+)+)(?:\s*[A-Za-z]\)?\)?)?", re.IGNORECASE)
_STATUS_RE = re.compile(r"\b(Acceptable|Target|Defect|Non[- ]?conforming)\b", re.IGNORECASE)


def _bbox_to_xywh(b: Optional[Tuple[float, float, float, float]]) -> List[Optional[float]]:
    if not b:
        return [None, None, None, None]
    x0, y0, x1, y1 = b
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    if w <= 0 or h <= 0:
        return [None, None, None, None]
    return [float(x0), float(y0), float(w), float(h)]


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    aa = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    bb = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return float(inter / (aa + bb - inter + 1e-6))


def _center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = b
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _dist(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay = _center(a); bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def _normalize_status(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = _STATUS_RE.search(s)
    if not m:
        return None
    word = m.group(1)
    # Normalize common spellings
    if re.match(r"non", word, re.IGNORECASE):
        return "Nonconforming"
    return word.capitalize()


# ========================
# Document JSON Assembler
# ========================
def assemble_document_json(pdf_path: str, doc_id: str,
                           *, out_json_dir: str = "out_json") -> Dict[str, Any]:
    """
    Runs the full pipeline over ALL pages, assembles the target JSON, and saves it.
    Missing fields are filled with clear PENDING placeholders as requested.
    """
    # Lazy imports to avoid circulars when this file is imported elsewhere
    from layout_detect import detect_layout
    from figure_association import associate_figures

    os.makedirs(out_json_dir, exist_ok=True)

    # -------- Pass 1: collect per-page layouts & artifacts --------
    pdf = fitz.open(pdf_path)
    num_pages = pdf.page_count
    pdf.close()

    page_layouts: List[Dict[str, Any]] = []
    page_figs_raw: List[List[Dict[str, Any]]] = []  # from associate_figures (one list per page)
    print(f"Processing {num_pages} pages from: {pdf_path}")

    for p in range(num_pages):
        rep = load_pdf_page(pdf_path, page_number=p)
        layout = detect_layout(rep, pdf_path=pdf_path, page_number=p)
        page_layouts.append(layout)

        # Associate figures using the current heuristic
        figs = associate_figures(layout)

        # -- refine legend linking to be bounded between this caption and the next one --
        #    and to favor same-column legends. We'll recompute legends here
        captions_on_page = [b for b in layout["blocks"] if b.get("type") == "caption"]
        captions_on_page.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        legends_on_page = [b for b in layout["blocks"] if b.get("type") == "legend_item"]
        legends_on_page.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        def _next_caption_y0(current_cap_bbox):
            y0 = current_cap_bbox[1]
            after = [c for c in captions_on_page if c["bbox"][1] > y0 + 1.0]
            return after[0]["bbox"][1] if after else float("inf")

        for f in figs:
            # find the caption bbox this figure came from (match by text)
            cap_text = f.get("caption") or ""
            cap_block = None
            for c in captions_on_page:
                if c.get("text") == cap_text:
                    cap_block = c
                    break
            if not cap_block:
                continue
            cx0, cy0, cx1, cy1 = cap_block["bbox"]
            next_y0 = _next_caption_y0(cap_block["bbox"])  # vertical upper bound for legends under this caption

            # Select legend lines: below the caption, above next caption, same column-ish
            legends_near = []
            for lg in legends_on_page:
                lx0, ly0, lx1, ly1 = lg["bbox"]
                if ly0 <= cy1:
                    continue
                if ly0 >= next_y0:
                    continue
                # same column / horizontal band overlap with caption box
                horiz_overlap = max(0.0, min(lx1, cx1) - max(lx0, cx0))
                if horiz_overlap <= 0:
                    # still allow if left edges are close (ragged columns)
                    if abs(lx0 - cx0) > 80:
                        continue
                legends_near.append((ly0, lg["text"]))
            legends_near.sort(key=lambda t: t[0])
            f["legend"] = [t[1] for t in legends_near]

        page_figs_raw.append(figs)

    # -------- Pass 2: merge image_uri and status per figure, collect tables --------
    all_figures: List[Dict[str, Any]] = []
    all_tables: List[Dict[str, Any]] = []

    for p, (layout, figs_raw) in enumerate(zip(page_layouts, page_figs_raw)):
        # candidate map for image_uri merge
        cand_list = layout.get("figure_candidates", [])
        # status blocks on this page
        status_blocks = [b for b in layout["blocks"] if b.get("type") == "status"]
        # captions (useful for figure <-> caption geometry)
        caption_blocks = [b for b in layout["blocks"] if b.get("type") == "caption"]

        for f in figs_raw:
            # fid
            fid = f.get("fid") or None
            if not fid or fid == "unknown":
                # try to extract id from caption text
                cap_txt = f.get("caption") or ""
                m = _FID_RE.search(cap_txt)
                fid = m.group(1) if m else None
            if not fid:
                fid = "PENDING:figure id"

            # bbox as chosen by associate_figures; if none, try to borrow from the closest candidate above its caption
            bbox = f.get("bbox")
            if not bbox:
                # find caption bbox
                cap_txt = f.get("caption") or ""
                matching_caps = [c for c in caption_blocks if c.get("text") == cap_txt]
                if matching_caps and cand_list:
                    cbox = matching_caps[0]["bbox"]
                    # choose the nearest candidate above this caption
                    above = []
                    for cand in cand_list:
                        r = tuple(cand.get("bbox", ()))
                        if not r:
                            continue
                        if r[3] <= cbox[1] + 60:  # allow tiny slack below
                            d = abs(cbox[1] - r[3])
                            above.append((d, r))
                    if above:
                        above.sort(key=lambda t: t[0])
                        bbox = above[0][1]

            # merge image_uri if a candidate has large IoU with this bbox
            image_uri = None
            if bbox:
                for cand in cand_list:
                    cb = cand.get("bbox")
                    if not cb:
                        continue
                    try:
                        if _iou(tuple(bbox), tuple(cb)) >= 0.70:
                            image_uri = cand.get("image_uri")
                            break
                    except Exception:
                        pass

            if not image_uri:
                image_uri = "PENDING:figure image crop"

            # status: pick the nearest status block to the caption or figure
            status_val = None
            try:
                ref_box = bbox if bbox else None
                # prefer distance from caption if available
                cap_txt = f.get("caption") or ""
                for c in caption_blocks:
                    if c.get("text") == cap_txt:
                        ref_box = c.get("bbox")
                        break
                if status_blocks and ref_box:
                    best = min(status_blocks, key=lambda b: _dist(tuple(b["bbox"]), tuple(ref_box)))
                    status_val = _normalize_status(best.get("text"))
            except Exception:
                status_val = None

            # Legend already refined above; ensure it's a list
            legend_list = f.get("legend") or []

            # Build final figure object
            fig_obj = {
                "fid": fid,
                "caption": f.get("caption") or "PENDING:caption OCR",
                "status": status_val or "PENDING:status not detected",
                "image_uri": image_uri,
                "bbox": _bbox_to_xywh(bbox),
                "hotspots": [
                    {"label": "PENDING", "text": "PENDING:hotspot extraction", "bbox": [None, None, None, None]}
                ],
                "legend": legend_list,
                "page": p + 1,  # 1-based display index
            }
            all_figures.append(fig_obj)

        # tables (already strict caption-matched in layout)
        for tc in layout.get("table_candidates", []) or []:
            tbox = tc.get("bbox")
            # parse tid from caption_text if present
            tid = None
            cap = tc.get("caption_text") or ""
            m = _TID_RE.search(cap)
            if m:
                tid = m.group(1)
            tab_obj = {
                "tid": tid or "PENDING:table id",
                "csv_uri": "PENDING:table CSV",
                "bbox": _bbox_to_xywh(tbox),
                "page": p + 1,  # 1-based
            }
            all_tables.append(tab_obj)

    # -------- Pass 3: build sections and attach figures --------
    sections_out: List[Dict[str, Any]] = []

    # Collect all section headers across pages
    headers: List[Dict[str, Any]] = []
    for p, layout in enumerate(page_layouts):
        for b in layout.get("blocks", []):
            if b.get("type") == "section_header":
                raw = b.get("text") or ""
                sid = "PENDING:section id"
                title = "PENDING:section title"
                m = _SID_RE.match(raw)
                if m:
                    sid = m.group(1)
                    title = m.group(2).strip() or "PENDING:section title"
                headers.append({
                    "sid": sid,
                    "title": title,
                    "bbox": tuple(b.get("bbox", (None, None, None, None))),
                    "page": p + 1,
                })

    # Sort headers in reading order
    headers.sort(key=lambda h: (h["page"], h["bbox"][1] if h["bbox"][1] is not None else 1e9, h["bbox"][0] if h["bbox"][0] is not None else 1e9))

    # Helper: for an artifact (fig/table), find nearest preceding header
    def _nearest_header(page: int, y0: Optional[float]) -> Optional[int]:
        idx = None
        best_key = None
        for i, h in enumerate(headers):
            hp = h["page"]
            hy0 = h["bbox"][1] if h["bbox"][1] is not None else -1e9
            if hp < page or (hp == page and (y0 is None or hy0 <= y0)):
                key = (hp, hy0)
                if (best_key is None) or (key > best_key):
                    best_key = key
                    idx = i
        return idx

    # Seed empty sections from headers
    for h in headers:
        sections_out.append({
            "sid": h["sid"],
            "title": h["title"],
            "text": "PENDING:section text assembly",
            "pages": sorted({h["page"]}),
            "figures": [],
        })

    # Attach figures to nearest header; collect pages into section pages
    unsectioned_figs: List[Dict[str, Any]] = []
    for fig in all_figures:
        page = fig.get("page") or 1
        # y0 from bbox
        b = fig.get("bbox") or [None, None, None, None]
        y0 = b[1]
        idx = _nearest_header(page, y0)
        if idx is None:
            unsectioned_figs.append(fig)
        else:
            sections_out[idx]["figures"].append({
                # conform to the exact figure schema under sections
                "fid": fig["fid"],
                "caption": fig["caption"],
                "status": fig["status"],
                "image_uri": fig["image_uri"],
                "bbox": fig["bbox"],
                "hotspots": fig["hotspots"],
                "legend": fig.get("legend", []),
            })
            # update pages set
            pages_set = set(sections_out[idx]["pages"]) | {page}
            sections_out[idx]["pages"] = sorted(pages_set)

    # If any figures had no section, create a synthetic placeholder section
    if unsectioned_figs:
        pages = sorted({f.get("page") or 1 for f in unsectioned_figs})
        sections_out.append({
            "sid": "PENDING:unsectioned",
            "title": "PENDING:unsectioned content",
            "text": "PENDING:section text assembly",
            "pages": pages,
            "figures": [
                {
                    "fid": f["fid"],
                    "caption": f["caption"],
                    "status": f["status"],
                    "image_uri": f["image_uri"],
                    "bbox": f["bbox"],
                    "hotspots": f["hotspots"],
                    "legend": f.get("legend", []),
                }
                for f in unsectioned_figs
            ],
        })

    # -------- Pass 4: derived tags --------
    derived_tags: List[str] = []
    # status tags from any figure
    status_vals = set()
    for s in sections_out:
        for f in s.get("figures", []):
            norm = _normalize_status(f.get("status"))
            if norm:
                status_vals.add(norm.lower())
    for st in sorted(status_vals):
        derived_tags.append(f"status:{st}")

    if any(s.get("figures") for s in sections_out):
        derived_tags.append("topic:has-figures")
    if all_tables:
        derived_tags.append("topic:has-tables")

    # Placeholders for future classifiers
    derived_tags.append("PENDING:class extraction")
    derived_tags.append("PENDING:topic classification")

    # -------- Assemble final doc --------
    final_json: Dict[str, Any] = {
        "doc_id": doc_id or "PENDING:doc id",
        "sections": sections_out,
        "tables": all_tables,
        "derived_tags": derived_tags,
    }

    # -------- Save --------
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_json_dir, f"{pdf_stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    # Optional compact version
    out_min = os.path.join(out_json_dir, f"{pdf_stem}.min.json")
    with open(out_min, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, separators=(",", ":"))

    # Light log summary
    total_figs = sum(len(s.get("figures", [])) for s in sections_out)
    print("===============================================")
    print(f"Saved JSON:       {os.path.abspath(out_path)}")
    print(f"Saved compact:    {os.path.abspath(out_min)}")
    print(f"Pages:            {num_pages}")
    print(f"Sections:         {len(sections_out)}")
    print(f"Figures:          {total_figs} ")
    print(f"Tables:           {len(all_tables)} ")
    print("===============================================")
    print()
    return final_json


# ==================
# One-command runner
# ==================
if __name__ == "__main__":
    # ---- Configure your source PDF here (no CLI args needed) ----
    # Pick ONE of these or set your own path.
    pdf_path = r"1-7/2.0/IPC_0012_SCH.pdf" # many sections
    pdf_path = r"1-7/1.0/IPC_0002_SCH.pdf" # many figures
    pdf_path = r"1-7/5.0/IPC_0055_SCH.pdf" # flat figures
    pdf_path = r"1-7/4.0/IPC_0018_SCH.pdf" # standard test
    pdf_path = r"1-7/4.0/IPC_0016_SCH.pdf" # 2 sections side by side
    pdf_path = r"1-7/1.0/IPC_0010_SCH.pdf" # big table
    pdf_path = r"1-7/5.0/IPC_0036_SCH.pdf" # weird figure
    pdf_path = r"1-7/1.0/IPC_0004_SCH.pdf" # 2 tables

    pdf_paths = [
        r"1-7/2.0/IPC_0012_SCH.pdf",  # many sections
        r"1-7/1.0/IPC_0002_SCH.pdf",  # many figures
        r"1-7/5.0/IPC_0055_SCH.pdf",  # flat figures
        r"1-7/4.0/IPC_0018_SCH.pdf",  # standard test
        r"1-7/4.0/IPC_0016_SCH.pdf",  # 2 sections side by side
        r"1-7/1.0/IPC_0010_SCH.pdf",  # big table
        r"1-7/5.0/IPC_0036_SCH.pdf",  # weird figure
        r"1-7/1.0/IPC_0004_SCH.pdf",  # 2 tables
    ]

    # Set the document ID explicitly (do NOT infer). Update as needed.
    DOC_ID = "IPC-A-610J"  # or set to "PENDING:doc id" and fill later

    for pdf_path in pdf_paths:
        # Run the full pipeline (loads pages, runs layout/association, assembles JSON, saves it)
        assemble_document_json(pdf_path, DOC_ID)
