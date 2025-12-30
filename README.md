# IPC scanner
Parses IPC documents to extract JSON data, figures, and tables.

## Output
- `figures/`, `tables/`: extracted figure/table crops
- `out_json/`: assembled document JSON (pretty + compact)
- `layouts/`: PDF overlays with rectangles around detected regions

## Setup
- `pip install -r requirements.txt`
- Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki and set `TESSERACT_CMD` in `.env` (see `env.example`).

## How to run
- Set your PDF paths in `first_step_loader.py`.
- Run:
  ```bash
  python first_step_loader.py
  ```

## How does it work
- `first_step_loader.py` loads each PDF page with PyMuPDF, recording text blocks and image rectangles. Its `assemble_document_json` orchestrates the full run over every page and writes both prettified and compact JSON to `out_json/`.
- `layout_detect.py` merges native PDF text with Tesseract OCR (path comes from `TESSERACT_CMD` in `.env`) because native PDF extraction can miss lines; it labels blocks (captions, section headers, status, legends, body), finds figure/table regions with OpenCV, drops text that sits inside those regions, saves crops to `figures/` and `tables/`, and writes per-page layout overlays plus JSON to `layouts/`.
- `figure_association.py` links each caption to the nearest figure candidate above it, pulls nearby legend lines, and attaches a status if present.
- In `assemble_document_json`, figures get refined (IDs from captions, image URIs merged from saved crops, status nearest to caption/figure). Tables are added from matched table candidates. Section headers (matched via numbering like `^(\d+\.\\d+(?:\.\\d+)*)`) become the parent containers, and figures are attached to the nearest preceding header; anything unsectioned is grouped under a placeholder section.
- Derived tags (e.g., `topic:has-figures`) and clear `PENDING:` placeholders are added for fields not yet extracted so downstream consumers can spot missing data.

## Todo
- Refactor from loading specific file paths to loading all files in a specified folder.
- Improve reliability of OCR.
- Auto-fill `doc_id` instead of hardcoding it in `first_step_loader.py`.
- Extract section body text (now `PENDING:section text assembly`).
- Extract figure hotspots (labels/text/bboxes) instead of placeholder `PENDING` entries.
- Parse and save tables as CSVs instead of just saving them as images (`csv_uri` now `PENDING`); planned tools: Camelot/Tabula for PDFs, visual table detector for images.
- Detect numbered hotspots inside figure crops (contour/shape filter + OCR of digits) and link them to legend map.
- Fill classification tags via rules (e.g., class 1/2/3 → `class:1,2,3`, hardware keywords → `topic:hardware-sequence`) instead of the current `PENDING` placeholders.

## Problems to fix
- Figure detection for `1-7/5.0/IPC_0036_SCH.pdf` is problematic: normal text starting with “Figure...” is mistaken for figure captions, leading to extra figures.
- Overlap between default PDF OCR and Tesseract OCR.
