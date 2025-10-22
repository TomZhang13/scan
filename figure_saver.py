# figure_saver.py
import os
from typing import List, Dict, Any
import fitz  # PyMuPDF


def save_figures_from_pdf(pdf_path: str,
                          page_number: int,
                          figures: List[Dict[str, Any]],
                          out_dir: str = "figures") -> List[Dict[str, Any]]:
    """Crop each figure bbox from the given page and save as PNG in out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    updated = []
    for fig in figures:
        bbox = fig.get("bbox")
        fid = fig.get("fid", "unknown")

        if not bbox:
            continue  # skip missing bbox

        rect = fitz.Rect(bbox)
        pix = page.get_pixmap(clip=rect, dpi=150)
        out_path = os.path.join(out_dir, f"fig_{fid}.png")
        pix.save(out_path)

        fig["image_uri"] = os.path.abspath(out_path)
        updated.append(fig)

    doc.close()
    return updated
