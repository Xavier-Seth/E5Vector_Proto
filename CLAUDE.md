# CLAUDE.md — Project Instructions

## Project: E5Vector_Proto (word2vec-clean)

Flask REST API that performs OCR on uploaded documents, then classifies them into Philippine government HR/administrative document types using sentence embeddings and prototype centroid matching.

---

## Architecture

```
app.py          — Flask routes + classification logic
OcrService.py   — Multi-format text extraction (image, PDF, DOCX, Excel, TXT)
prototypes_e5.npz — Serialized prototype centroids (auto-generated, do not hand-edit)
training_samples/<ClassName>/*.txt — One text file per training example
templates/index.html — Simple web UI for text classification
tests/          — pytest test suite
```

### Classification flow

1. File upload → `OcrService.extract_text()` → raw text
2. `clean_text()` normalizes whitespace/artifacts
3. Word count guard: `< 50 words` → return `Uncategorized` immediately (avoids noisy embeddings)
4. `embed_text()` calls `intfloat/multilingual-e5-base` (normalized)
5. Cosine similarity against all prototype centroids (`PROTOTYPES` dict, loaded from `prototypes_e5.npz`)
6. Top score `< 0.90` → `Uncategorized`; otherwise return winning class label

### Prototype rebuild flow

`POST /rebuild-prototypes` → reads every `training_samples/<Class>/*.txt`, embeds each, averages vectors per class, saves to `prototypes_e5.npz`.

---

## Document classes

| Class dir | Document type |
|---|---|
| `Appointment Form` | CSC appointment papers |
| `Certification of Assumption to Duty` | Assumption certificate |
| `Daily Time Record` | DTR / timesheet |
| `ICS` | Inventory Custodian Slip |
| `NOSA` | Notice of Step Advancement |
| `NOSI` | Notice of Salary Increase |
| `Oath of Office` | Oath form |
| `Personal Data Sheet` | CSC Form 212 |
| `RIS` | Requisition and Issue Slip |
| `Transcript of Records` | Academic TOR |
| `Travel Order` | Travel order memo |
| `Work Experience Sheet` | CS Form 212 attachment |

---

## Key constants (app.py)

| Constant | Value | Purpose |
|---|---|---|
| `EMBED_MODEL_NAME` | `intfloat/multilingual-e5-base` | Embedding model |
| `SIMILARITY_THRESHOLD` | `0.90` | Min cosine similarity to classify (else Uncategorized) |
| `WORD_COUNT_MIN` | `50` | Min words to attempt classification |
| `PROTO_FILE` | `prototypes_e5.npz` | Cached prototypes |

---

## API endpoints

| Method | Route | Input | Output |
|---|---|---|---|
| GET | `/` | — | Health check |
| POST | `/rebuild-prototypes` | — | Rebuilds + saves prototypes |
| POST | `/classify` | `{"text": "..."}` | Classification result JSON |
| POST | `/extract-and-classify` | multipart `file` | OCR + classify result JSON |

### Response shape

```json
{
  "main_category": "Teacher Profile",
  "subcategory": "<ClassName or Uncategorized>",
  "similarity": 0.9234,
  "method": "E5 + prototype centroid",
  "threshold": 0.90,
  "too_short": false,
  "word_count": 123,
  "min_words_required": 50,
  "candidates": [{"label": "...", "sim": 0.92}, ...],
  "text_preview": "..."
}
```

---

## Running the project

```powershell
# Activate venv (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dev server
python app.py         # → http://127.0.0.1:5000

# Rebuild prototypes after adding training samples
curl -X POST http://127.0.0.1:5000/rebuild-prototypes

# Run tests
pytest tests/
```

### System dependencies (not in pip)

- **Tesseract OCR** — must be on PATH (`tesseract`)
- **Poppler** — required for PDF-to-image conversion; add `bin/` to PATH on Windows

---

## Adding a new document class

1. Create `training_samples/<NewClass>/` directory
2. Add ≥10 representative `.txt` files (at least 10 words each; 50+ recommended)
3. `POST /rebuild-prototypes` to recompute centroids

---

## Tests

```powershell
pytest tests/ -v
```

Key test files:
- `tests/test_routes.py` — route smoke tests and short-text guard
- `tests/test_decision.py` — `_decide_label` logic
- `tests/test_ocr.py` / `test_ocr_helpers.py` — OcrService methods
- `tests/test_helpers.py` — utility functions

---

## Do not

- Manually edit `prototypes_e5.npz` — it's auto-generated binary
- Commit files in `temp_uploads/` — ephemeral upload scratch dir
- Lower `SIMILARITY_THRESHOLD` below `0.85` without testing — causes false positives on short docs
