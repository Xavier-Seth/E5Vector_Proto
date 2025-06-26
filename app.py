from flask import Flask, request, jsonify, render_template
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import traceback
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
import os

from OcrService import OcrService  # ✅ Import the class you created

# 🧠 Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔧 App + OCR Init
app = Flask(__name__, static_folder="static")
ocr = OcrService()

print("Loading Word2Vec model... Please wait.")
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
print("✅ Word2Vec model loaded.")

# 🧠 Rule-based Keywords
teacher_rules = {
    "Personal Data Sheet": ["personal data sheet"],
    "Work Experience Sheet": ["work experience sheet"],
    "Oath of Office": ["oath of office"],
    "Certification of Assumption to Duty": ["certification of assumption to duty"],
    "ICS": ["inventory custodian slip", "ics"],
    "RIS": ["requisition and issue slip", "ris"],
    "Transcript of Records": ["transcript of records", "official transcript of records"],
    "Appointment Form": ["cs fom no. 3-a", "you are hereb apponted"],
    "Daily Time Record": ["daily time record", "form 48", "civil service form no. 48", "dtr"],  # ✅ NEW
}

# 🧠 Vector-based Keywords
teacher_vectors = {
    "Personal Data Sheet": [
        "profile", "civil service", "personal", "family", "background",
        "education", "eligibility", "bio", "birthdate", "contact", "record"
    ],
    "Work Experience Sheet": [
        "experience", "work", "position", "responsibility", "agency",
        "employment", "career", "job", "roles", "functions", "duration", "achievements",
        "employment background", "supervisor", "accomplishments", "past duties"
    ],
    "Oath of Office": [
        "oath", "faithfully", "swear", "republic", "obey", "allegiance",
        "constitution", "loyalty", "duties", "responsibilities", "voluntarily", "so help me god"
    ],
    "Certification of Assumption to Duty": [
        "assumed", "certification", "assumption", "duties", "cs form no. 4"
    ],
    "ICS": [
        "inventory", "custodian", "slip", "ics", "unit cost", "total cost", "fund cluster",
        "received from", "inventory item", "estimated useful life"
    ],
    "RIS": [
        "requisition", "issue", "slip", "ris", "stock number", "requisitioned by",
        "approved by", "purpose", "item description", "quantity requested"
    ],
    "Transcript of Records": [
        "transcript", "official transcript", "final grade", "units of credit",
        "course name", "descriptive title", "university registrar",
        "term", "semester", "date graduated", "degree", "marks", "rating",
        "entrance data", "bachelor of science", "institute of architecture",
        "grading system", "remarks", "place of birth", "date conferred"
    ],
    "Appointment Form": [
        "appointed", "appointment", "position", "salary", "civil service",
        "plantilla", "appointing officer", "original", "promotion", "vice",
        "date of signing", "status", "job title", "authorized", "cs form no. 33-a"
    ],
    "Daily Time Record": [
        "arrival", "departure", "under time", "in-charge",
        "official hours", "civil service form no. 48", "daily record", "dtr", "attendance"
    ]
}

SIMILARITY_THRESHOLD = 0.30

# 🧼 Tokenizing and Scoring
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in stop_words and w in model]

def average_vector(words):
    freq = Counter(words)
    total = sum(freq.values())
    vectors, weights = [], []
    for word in words:
        if word in model:
            vectors.append(model[word])
            weights.append(freq[word] / total)
    if not vectors:
        return np.ones((1, model.vector_size)) * 1e-10
    return np.average(vectors, axis=0, weights=weights).reshape(1, -1)

def contains_strict_keyword(text, keyword):
    return re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)

# 🏠 Test UI
@app.route("/")
def index():
    return render_template("index.html")

# 🧠 Manual Text Classification (UI testing)
@app.route("/classify", methods=["POST"])
def classify_text():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Empty input text"}), 400

        tokens = tokenize(text)

        for subcategory, keywords in teacher_rules.items():
            for keyword in keywords:
                if contains_strict_keyword(text, keyword):
                    return jsonify({
                        "main_category": "Teacher Profile",
                        "subcategory": subcategory,
                        "method": "rule-based",
                        "text": text
                    })

        doc_vector = average_vector(tokens)
        best_match = "Uncategorized"
        highest_score = 0.0

        for subcategory, keywords in teacher_vectors.items():
            cat_vector = average_vector(keywords)
            score = cosine_similarity(doc_vector, cat_vector)[0][0]
            if score > highest_score:
                highest_score = score
                best_match = subcategory

        return jsonify({
            "main_category": "Teacher Profile",
            "subcategory": best_match if highest_score >= SIMILARITY_THRESHOLD else "Uncategorized",
            "similarity": float(round(highest_score, 4)),
            "method": "vector-based",
            "text": text
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# 🔁 File Upload + OCR + Classification
@app.route("/extract-and-classify", methods=["POST"])
def extract_and_classify():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = file.filename

        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        text = ocr.extract_text(temp_path)
        text = text.replace("nan", "").replace("\x0c", "").strip()
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        print("📄 OCR Extracted Text:")
        print(text)

        tokens = tokenize(text)
        if not tokens:
            os.remove(temp_path)
            return jsonify({
                "main_category": "Teacher Profile",
                "subcategory": "Uncategorized",
                "method": "no match (empty OCR text)",
                "similarity": 0.0,
                "text": text
            })

        for subcategory, keywords in teacher_rules.items():
            for keyword in keywords:
                if contains_strict_keyword(text, keyword):
                    os.remove(temp_path)
                    return jsonify({
                        "main_category": "Teacher Profile",
                        "subcategory": subcategory,
                        "method": "rule-based",
                        "text": text
                    })

        doc_vector = average_vector(tokens)
        best_match = "Uncategorized"
        highest_score = 0.0

        for subcategory, keywords in teacher_vectors.items():
            cat_vector = average_vector(keywords)
            score = cosine_similarity(doc_vector, cat_vector)[0][0]
            if score > highest_score:
                highest_score = score
                best_match = subcategory

        os.remove(temp_path)

        return jsonify({
            "main_category": "Teacher Profile",
            "subcategory": best_match if highest_score >= SIMILARITY_THRESHOLD else "Uncategorized",
            "similarity": float(round(highest_score, 4)),
            "method": "vector-based",
            "text": text
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# 🚀 Run Server
if __name__ == "__main__":
    app.run(port=5000)
