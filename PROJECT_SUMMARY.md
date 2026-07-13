# E5Vector_Proto — Project Summary

## What This Project Does

REST API service that automatically identifies the type of a Philippine government HR/administrative document. A user uploads a file (or pastes text); the system extracts text via OCR, embeds it using a multilingual sentence transformer, and returns the document category (e.g. "Daily Time Record", "Personal Data Sheet", "Oath of Office") along with a confidence score.

## Who It's Built For

School administrative staff who handle teacher personnel documents. Instead of manually sorting stacks of scanned paperwork, they upload a file and the system tells them what it is — reducing manual tagging effort in a document management workflow.

## Key Features

- **Multi-format OCR** — accepts JPG/PNG/TIFF images, PDFs (via Poppler), DOCX, XLS/XLSX, and plain TXT
- **Semantic classification** — uses `intfloat/multilingual-e5-base` embeddings rather than keyword matching; handles OCR noise and varied phrasings
- **Prototype centroid model** — each document class is represented by the average embedding of its training samples; no GPU or fine-tuning required
- **Short-text guard** — rejects documents with fewer than 50 words as "Uncategorized" to avoid confident wrong predictions on bad scans
- **Similarity threshold** — cosine similarity must reach 0.90 to classify; anything below returns "Uncategorized"
- **Rebuild endpoint** — add new training samples and hit `POST /rebuild-prototypes` to update the model instantly; no redeployment needed
- **12 document classes** — Appointment Form, Assumption to Duty, Daily Time Record, ICS, NOSA, NOSI, Oath of Office, Personal Data Sheet, RIS, Transcript of Records, Travel Order, Work Experience Sheet

## Tech Stack

| Layer             | Technology                                                |
| ----------------- | --------------------------------------------------------- |
| Web framework     | Flask 3.x (Python)                                        |
| Embeddings        | `intfloat/multilingual-e5-base` via Sentence Transformers |
| Similarity        | scikit-learn cosine_similarity                            |
| Numerics          | NumPy (prototype storage as `.npz`)                       |
| OCR engine        | Tesseract (pytesseract)                                   |
| PDF rasterization | Poppler + pdf2image                                       |
| DOCX parsing      | python-docx                                               |
| Excel parsing     | pandas + openpyxl + xlrd                                  |
| Testing           | pytest                                                    |
| Runtime           | Python 3.x, virtualenv                                    |

## System Architecture

```
Client (HTTP)
      │
      ▼
Flask REST API (app.py)
      │
      ├── POST /extract-and-classify
      │       │
      │       ▼
      │   OcrService.extract_text()
      │   (dispatch by file extension)
      │       │ Tesseract / python-docx / pandas / plain read
      │       ▼
      │   clean_text()  →  word count guard (< 50 → Uncategorized)
      │       │
      │       ▼
      │   embed_text()  →  multilingual-e5-base (normalized L2)
      │       │
      │       ▼
      │   cosine_similarity(doc_vec, prototype_centroids)
      │       │
      │       ▼
      │   _decide_label()  →  threshold check (≥ 0.90 to classify)
      │       │
      │       └──► JSON response { subcategory, similarity, candidates }
      │
      └── POST /rebuild-prototypes
              │
              ▼
          Read training_samples/<Class>/*.txt
          embed each → average per class → save prototypes_e5.npz
```

Prototypes are loaded into memory at startup (`PROTOTYPES` dict). Rebuild is the only write path; inference is fully read-only after load.

## Notable Technical Decisions

**Prototype centroid instead of a trained classifier** — With ~20 examples per class, training a neural classifier would overfit. Averaging embeddings into a centroid per class is zero-shot-friendly, interpretable, and easy to update: add samples, hit rebuild.

**`intfloat/multilingual-e5-base` over Word2Vec** — Despite the repo name, the final implementation uses E5. E5 produces contextualized sentence-level embeddings that are robust to OCR artifacts and handle Filipino/English mixed text better than static word vectors.

**0.90 cosine threshold** — Government forms share a lot of boilerplate. A high threshold prevents a DTR from being mislabeled as a PDS because both mention an employee's name and office. "Uncategorized" is a safer fallback than a confident wrong label.

**50-word minimum** — A badly scanned page may yield 10 OCR tokens. Embedding such text and comparing it to a centroid produces unreliable similarity values. Rejecting it outright forces the user to re-scan rather than silently misfiling the document.

**`prototypes_e5.npz` cache** — The embedding model takes several seconds to load. Serializing computed centroids means inference restarts skip the rebuild step while still surviving server restarts.

## Challenges & What I Learned

One of the hardest decisions in this project was choosing the right classification algorithm. I initially built the classifier using Word2Vec with hardcoded rules — meaning the categories and matching logic were manually written into the code. While it worked to some extent, it was rigid and difficult to maintain. I eventually replaced it entirely with the multilingual E5 vector model, which learns from real document samples instead of hardcoded rules. The model classifies documents by comparing them against actual snippets of real documents I collected, making it far more accurate and flexible. The biggest learning was that good classification doesn't come from writing more rules — it comes from giving the model better examples to learn from.

One of the hardest decisions was choosing the right similarity threshold. I initially tested lower values around 0.70–0.75, but the model would confidently accept documents that had nothing to do with the target categories — even documents completely outside the training set would get classified. That was a clear sign the threshold was too loose. After testing at 0.90, the accuracy improved significantly — the model became much more selective and only classified documents it was genuinely confident about, returning "Uncategorized" for everything else rather than making a wrong guess.
I also moved away from Word2Vec to the E5 multilingual model because Word2Vec relied entirely on hardcoded rules that were brittle and hard to maintain. E5 learns from real document snippets, which made it far more flexible and accurate — especially on OCR-scanned government forms that have a lot of noise and mixed Filipino/English text.
If I had more time, I would build a much larger training set and explore fine-tuning a model specifically on Philippine government documents to push accuracy even higher. I would also consider a proper database backend for storing prototypes and training samples instead of flat files — something with more scalability as the document classes grow.

---

_Generated from source code analysis_
