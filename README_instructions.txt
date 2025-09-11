# E5Vector_Proto

Flask-based OCR + Document Classification service using **Sentence-Transformers (E5)** and **prototype vectors**.  
Supports classification of PDFs, images, Word, Excel, and TXT files into categories such as *PDS, DTR, ICS, RIS, TOR, etc.*

---

## 🚀 Features
- OCR support for:
  - **Images** (JPG, PNG, TIFF, BMP)
  - **PDFs** (via Poppler + Tesseract)
  - **DOCX** (Word documents)
  - **XLS/XLSX** (Excel spreadsheets)
  - **TXT**
- Embedding + classification using **E5 multilingual model**
- Prototype-based centroid classification
- REST API endpoints:
  - `POST /rebuild-prototypes` → rebuilds prototypes from `training_samples/`
  - `POST /classify` → classify raw text
  - `POST /extract-and-classify` → upload a file and auto-classify

---

## 🛠 Setup

### 1. Clone the repo
```bash
git clone https://github.com/Xavier-Seth/E5Vector_Proto.git
cd E5Vector_Proto
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate it:
- **Windows (PowerShell):**
  ```bash
  .venv\Scripts\activate
  ```
- **Linux/Mac/Git Bash:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install system packages (required for OCR)
#### Ubuntu/Debian (VPS/WSL):
```bash
sudo apt update
sudo apt install -y tesseract-ocr poppler-utils
```

#### Windows:
- Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)  
- Install [Poppler for Windows](https://blog.alivate.com.au/poppler-windows/) and add the `bin/` folder to PATH  

---

## ▶️ Running the Service
```bash
python app.py
```

- App will run on [http://127.0.0.1:5000](http://127.0.0.1:5000)
- Example test:
  ```bash
  curl -X POST -F "file=@test/oath-of-office.pdf" http://127.0.0.1:5000/extract-and-classify
  ```

---

## 📂 Training Samples
- Store text snippets in `training_samples/<CategoryName>/*.txt`
- Run:
  ```bash
  curl -X POST http://127.0.0.1:5000/rebuild-prototypes
  ```
  to rebuild prototype vectors (`prototypes_e5.npz` will be saved).

---

## 📦 Production (optional)
Run with Gunicorn for better performance:
```bash
pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:5000 app:app
```
