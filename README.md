# 🩺 Patient Intelligence Console

An agentic AI application for managing patient records with natural language queries, built using the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).

## Features

- **SQLite Database** - Persistent patient record storage
- **AI-Powered Queries** - Ask natural language questions about patient records
- **Vector Search (FAISS)** - Semantic search over patient documents using OpenAI embeddings
- **OCR Support** - Extract text from scanned PDFs and images
- **Admin Panel** - Add, edit, and import patient data
- **Document Management** - Digital + handwritten doctor notes and documents

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/bonda108/PMS.git
cd PMS
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 3.1 Install system dependencies for OCR (optional)

For OCR functionality (scanning handwritten notes and PDFs):

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

### 4. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 5. Run the application

```bash
./run.sh
# Or manually:
python -m streamlit run app.py --server.port 8501
```

The app will be available at: **http://localhost:8501**

## Sample Data

The application includes 10 mock patient profiles with:
- Vitals (blood pressure, heart rate, temperature, fever status)
- Medical history entries
- Doctor notes (digital + handwritten)
- Patient documents

## Technology Stack

- **Streamlit** - Web interface
- **OpenAI Agents SDK** - Agentic AI framework
- **LangChain** - LLM orchestration
- **FAISS** - Vector similarity search
- **SQLite** - Local database
- **Pytesseract** - OCR for scanned documents

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Frontend                      │
├─────────────────────────────────────────────────────────┤
│                 OpenAI Agents SDK                        │
│    ┌─────────────────┐    ┌─────────────────────────┐   │
│    │  Patient Agent  │────│  Function Tools          │   │
│    │                 │    │  - get_patient_profile   │   │
│    │                 │    │  - search_patient_history│   │
│    └─────────────────┘    └─────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│        FAISS Vector Store    │    SQLite Database       │
│   (Patient Doc Embeddings)   │   (Patient Records)      │
└─────────────────────────────────────────────────────────┘
```

## License

MIT
