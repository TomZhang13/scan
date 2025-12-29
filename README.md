# IPC scanner
parses IPC documents, extracts json data, figures, and tables

## output
/figures /tables extracted figures and tables from the documents
/out_json json data from the documents
/layouts pdfs with rectangles around extracted data for debug use

## setup
pip install requirements.txt
download tesseract from https://github.com/UB-Mannheim/tesseract/wiki, put the path in .env

## how to run
put in the paths of the documents in first_step_loader.py
run `python first_step_loader.py`

## how does it work
- `first_step_loader.py` loads each PDF page with PyMuPDF, recording text blocks and image rectangles. Its `assemble_document_json` orchestrates the full run over every page and writes both prettified and compact JSON to `out_json/`.
- `layout_detect.py` merges native PDF text with Tesseract OCR (path comes from `TESSERACT_CMD` in `.env`) because native PDF extraction can miss lines; it labels blocks (captions, section headers, status, legends, body), finds figure/table regions with OpenCV, drops text that sits inside those regions, saves crops to `figures/` and `tables/`, and writes per-page layout overlays plus JSON to `layouts/`.
- `figure_association.py` links each caption to the nearest figure candidate above it, pulls nearby legend lines, and attaches a status if present.
- Back in `assemble_document_json`, figures get refined (IDs from captions, image URIs merged from saved crops, status nearest to caption/figure). Tables are added from matched table candidates. Section headers (matched via numbering like `^(\d+\.\d+(?:\.\d+)*)`) become the parent containers, and figures are attached to the nearest preceding header; anything unsectioned is grouped under a placeholder section.

## todo
refactor from loading specific file paths to loading all files in a specified folder
improve reliability of ocr
auto-fill `doc_id` instead of hardcoding it in `first_step_loader.py`
extract section body text (now `PENDING:section text assembly`)
extract figure hotspots (labels/text/bboxes) instead of placeholder `PENDING` entries
generate table CSVs (`csv_uri` now `PENDING`); use tools like Camelot/Tabula for PDFs, visual table detector for images
detect numbered hotspots inside figure crops (contour/shape filter + OCR of digits) and link them to legend map
fill classification tags via rules (e.g., class 1/2/3 → `class:1,2,3`, hardware keywords → `topic:hardware-sequence`) instead of the current `PENDING` placeholders

## problems to fix
figure detection for the document "1-7/5.0/IPC_0036_SCH.pdf" is problematic. More figures are detected than what's on the document. This is because normal texts that start with "Figure..." are mistaken for figure texts, and the program looks for figures to associate those "figure texts" with.
overlap between default pdf ocr and tessaract ocr
