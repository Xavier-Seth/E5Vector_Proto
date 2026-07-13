from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os, re, traceback
from pathlib import Path

# (keep your existing OcrService.py)
from OcrService import OcrService

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
TRAIN_DIR = Path("training_samples")
PROTO_FILE = Path("prototypes_e5.npz")

# classification gates
SIMILARITY_THRESHOLD = 0.90           # similarity threshold
WORD_COUNT_MIN = 50             # NEW: minimum words required to classify

# ──────────────────────────────────────────────────────────────────────────────
# Init
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
ocr = OcrService()
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# in-memory prototypes
PROTOTYPES = {}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.zeros((embedder.get_sentence_embedding_dimension(),), dtype=np.float32)
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)

def average_vectors(vectors):
    if not vectors:
        return None
    mat = np.vstack(vectors)
    return np.mean(mat, axis=0)

def word_count(s: str) -> int:
    return len((s or "").split())

# ──────────────────────────────────────────────────────────────────────────────
# Prototypes
# ──────────────────────────────────────────────────────────────────────────────
def save_prototypes(proto_dict):
    labels = list(proto_dict.keys())
    vecs = np.vstack([proto_dict[k] for k in labels])
    np.savez_compressed(PROTO_FILE, labels=np.array(labels, dtype=object), vectors=vecs)

def load_prototypes():
    global PROTOTYPES
    if PROTO_FILE.exists():
        data = np.load(PROTO_FILE, allow_pickle=True)
        labels = list(data["labels"])
        vectors = data["vectors"]
        PROTOTYPES = {lbl: vectors[i] for i, lbl in enumerate(labels)}
    else:
        PROTOTYPES = {}

def rebuild_prototypes():
    global PROTOTYPES
    PROTOTYPES = {}
    if not TRAIN_DIR.exists():
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
    for cdir in class_dirs:
        snippets = []
        for txt_path in sorted(cdir.glob("*.txt")):
            try:
                raw = txt_path.read_text(encoding="utf-8", errors="ignore")
                cleaned = clean_text(raw)
                if len(cleaned.split()) < 10:
                    continue
                snippets.append(cleaned)
            except Exception:
                continue

        if not snippets:
            continue

        vectors = [embed_text(s) for s in snippets]
        centroid = average_vectors(vectors)
        if centroid is not None:
            PROTOTYPES[cdir.name] = centroid

    if PROTOTYPES:
        save_prototypes(PROTOTYPES)

# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(s: str) -> str:
    s = (s or "")
    s = s.replace("\x0c", " ")
    s = s.replace("nan", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ──────────────────────────────────────────────────────────────────────────────
# Decision helper (threshold only)
# ──────────────────────────────────────────────────────────────────────────────
def _decide_label(text: str, sims: np.ndarray, labels: list):
    order = np.argsort(-sims)
    top_idx = int(order[0])
    top_label = labels[top_idx]
    top_score = float(sims[top_idx])

    if top_score < SIMILARITY_THRESHOLD:
        return "Uncategorized", top_score

    return top_label, top_score

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({"ok": True, "msg": "E5 prototype classifier running."})

@app.route("/rebuild-prototypes", methods=["POST"])
def api_rebuild():
    try:
        rebuild_prototypes()
        return jsonify({
            "status": "ok",
            "classes": list(PROTOTYPES.keys()),
            "note": f"Saved to {str(PROTO_FILE.resolve())}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "rebuild failed", "message": str(e)}), 500

@app.route("/classify", methods=["POST"])
def classify_text():
    try:
        data = request.get_json(silent=True) or {}
        text = clean_text(data.get("text", ""))

        if not text:
            return jsonify({"error": "Empty input text"}), 400

        # NEW: short-text guard
        wc = word_count(text)
        if wc < WORD_COUNT_MIN:
            return jsonify({
                "main_category": "Teacher Profile",
                "subcategory": "Uncategorized",
                "similarity": 0.0,
                "method": "E5 + prototype centroid",
                "threshold": SIMILARITY_THRESHOLD,
                "too_short": True,
                "word_count": wc,
                "min_words_required": WORD_COUNT_MIN,
                "text_preview": text[:400]
            })

        if not PROTOTYPES:
            load_prototypes()
        if not PROTOTYPES:
            return jsonify({"error": "No prototypes found. POST /rebuild-prototypes first."}), 400

        doc_vec = embed_text(text).reshape(1, -1)
        labels = list(PROTOTYPES.keys())
        centroids = np.vstack([PROTOTYPES[lbl] for lbl in labels])

        sims = cosine_similarity(doc_vec, centroids)[0]
        result_label, top_score = _decide_label(text, sims, labels)

        return jsonify({
            "main_category": "Teacher Profile",
            "subcategory": result_label,
            "similarity": round(top_score, 4),
            "method": "E5 + prototype centroid",
            "threshold": SIMILARITY_THRESHOLD,
            "too_short": False,
            "word_count": wc,
            "min_words_required": WORD_COUNT_MIN,
            "candidates": sorted(
                [{"label": lbl, "sim": float(s)} for lbl, s in zip(labels, sims)],
                key=lambda x: x["sim"], reverse=True
            )[:5],
            "text_preview": text[:400]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route("/extract-and-classify", methods=["POST"])
def extract_and_classify():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        filename = secure_filename(f.filename or "")
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400
        temp_dir = Path("temp_uploads"); temp_dir.mkdir(exist_ok=True, parents=True)
        temp_path = temp_dir / filename
        f.save(temp_path)

        text = ocr.extract_text(str(temp_path))
        text = clean_text(text)
        os.remove(temp_path)

        # If OCR produced nothing
        if not text:
            return jsonify({
                "main_category": "Teacher Profile",
                "subcategory": "Uncategorized",
                "method": "E5 + prototype centroid",
                "similarity": 0.0,
                "too_short": True,
                "word_count": 0,
                "min_words_required": WORD_COUNT_MIN,
                "text_preview": ""
            })

        # NEW: short-text guard
        wc = word_count(text)
        if wc < WORD_COUNT_MIN:
            return jsonify({
                "main_category": "Teacher Profile",
                "subcategory": "Uncategorized",
                "method": "E5 + prototype centroid",
                "similarity": 0.0,
                "too_short": True,
                "word_count": wc,
                "min_words_required": WORD_COUNT_MIN,
                "text_preview": text[:400]
            })

        if not PROTOTYPES:
            load_prototypes()
        if not PROTOTYPES:
            return jsonify({"error": "No prototypes found. POST /rebuild-prototypes first."}), 400

        doc_vec = embed_text(text).reshape(1, -1)
        labels = list(PROTOTYPES.keys())
        centroids = np.vstack([PROTOTYPES[lbl] for lbl in labels])
        sims = cosine_similarity(doc_vec, centroids)[0]
        result_label, top_score = _decide_label(text, sims, labels)

        return jsonify({
            "main_category": "Teacher Profile",
            "subcategory": result_label,
            "similarity": round(top_score, 4),
            "method": "E5 + prototype centroid",
            "threshold": SIMILARITY_THRESHOLD,
            "too_short": False,
            "word_count": wc,
            "min_words_required": WORD_COUNT_MIN,
            "candidates": sorted(
                [{"label": lbl, "sim": float(s)} for lbl, s in zip(labels, sims)],
                key=lambda x: x["sim"], reverse=True
            )[:5],
            "text_preview": text[:400]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_prototypes()
    app.run(port=5000, debug=True)
