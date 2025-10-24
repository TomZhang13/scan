# first_step_loader.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

# PDF deps
import fitz  # PyMuPDF

# Image fallback (optional)
from PIL import Image


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
        if block_type == 0 and text.strip():
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

if __name__ == "__main__":
    pdf_path = r"1-7/1.0/IPC_0002_SCH.pdf"
    pdf_path = r"1-7/4.0/IPC_0023_SCH.pdf"
    pdf_path = r"1-7/4.0/IPC_0018_SCH.pdf"
    pdf_path = r"1-7/4.0/IPC_0026_SCH.pdf"
    pdf_path = r"1-7/4.0/IPC_0027_SCH.pdf"
    pdf_path = r"1-7/1.0/IPC_0003_SCH.pdf"
    page_no = 0

    rep = load_pdf_page(pdf_path, page_number=page_no)

    # PASS pdf_path + page_number so detect_layout will save figure crops
    from layout_detect import detect_layout
    layout = detect_layout(rep, pdf_path=pdf_path, page_number=page_no)

    # Optional: build figure objects from the layout (uses figure_candidates)
    from figure_association import associate_figures
    figures = associate_figures(layout)

    from pprint import pprint
    pprint(figures)
    # show where crops were saved by detect_layout
    print("saved crops:", [fc.get("image_uri") for fc in layout.get("figure_candidates", [])])
