# figure_association.py
from typing import Dict, Any, List
import re

def associate_figures(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Link each caption to its nearby legend_items and figure_candidate boxes.
    Simple geometry + regex approach.
    """
    blocks = layout["blocks"]
    figs_out: List[Dict[str, Any]] = []

    # build quick lists
    captions = [b for b in blocks if b["type"] == "caption"]
    legends  = [b for b in blocks if b["type"] == "legend_item"]
    statuses = [b for b in blocks if b["type"] == "status"]

    for cap in captions:
        text = cap["text"]
        m = re.search(r"Figure\s+(\d+-\d+)", text, re.IGNORECASE)
        fid = m.group(1) if m else "unknown"
        x0, y0, x1, y1 = cap["bbox"]

        # pick nearest figure candidate vertically overlapping the caption
        candidate = None
        min_dy = 1e9
        for fc in layout["figure_candidates"]:
            fx0, fy0, fx1, fy1 = fc["bbox"]
            if fy1 <= y0:  # figure above caption
                dy = y0 - fy1
                if dy < min_dy:
                    min_dy = dy
                    candidate = fc

        # collect legend lines below this caption until next caption
        legends_near = []
        for lg in legends:
            lx0, ly0, lx1, ly1 = lg["bbox"]
            if ly0 < y0:  # below caption
                # same column (roughly)
                if abs(lx0 - x0) < 50:
                    legends_near.append(lg["text"])
        legends_near = sorted(set(legends_near))

        # simple status pick (first on page)
        status = statuses[0]["text"] if statuses else None

        figs_out.append({
            "fid": fid,
            "caption": text,
            "status": status,
            "bbox": candidate["bbox"] if candidate else None,
            "legend": legends_near,
            "hotspots": [],         # placeholder for later step
            "image_uri": None,      # placeholder for cropped image later
        })

    return figs_out
