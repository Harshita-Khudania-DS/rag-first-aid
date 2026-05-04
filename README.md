# First Aid Assistant — RAG + Local LLM (Offline)

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) based First Aid Assistant that answers first-aid queries strictly using a custom dataset and a local LLM without internet access.

The system retrieves the most relevant first-aid cases using vector similarity search and sends only those retrieved cases to the LLM for response generation.

The LLM does not rely on its own memory. It generates answers only from retrieved context, ensuring reliable, grounded, and explainable responses.

---

## Key Features

- Fully offline system (No internet required)
- Uses 445+ first-aid cases from custom JSON dataset
- Semantic search using Hugging Face MiniLM embeddings
- Fast vector similarity retrieval using FAISS
- Local LLM inference using Ollama (`phi3:mini`)
- Strict JSON structured medical responses
- Reduced hallucination with prompt-bound generation
- Handles mixed emergency scenarios

---

## Tech Stack

- Python
- Hugging Face MiniLM (`all-MiniLM-L6-v2`)
- FAISS (Vector Database)
- Ollama
- phi3:mini (Local LLM)
- LangChain
- JSON Knowledge Base

---

## Project Structure

```text
rag-first-aid/
│
├── first_aid_data.json
├── embed.py
├── rag.py
├── main.py
├── README.md
│
├── faiss_index/   (not uploaded)
├── models/        (not uploaded)
└── venv/          (not uploaded)
````

---

## RAG Architecture

Classic Retrieve-Then-Read RAG

```
User Query
↓
MiniLM Embedding
↓
FAISS Top-K Search
↓
Retrieve Relevant Cases
↓
Context-Based Prompt
↓
Ollama (phi3:mini)
↓
Final JSON First-Aid Response
```

---

## How to Run

### First Time Setup / After JSON Change

```
venv\Scripts\activate
Remove-Item -Recurse -Force faiss_index
python embed.py
python rag.py
```

### Normal Run

```
venv\Scripts\activate
python rag.py
```

---

## Common Fix

If JSON data is updated:

Delete `faiss_index` → Run `embed.py` → Run `rag.py`

---

## Author

Harshita Khudania
BSc Data Science and Statistics Student
CHRIST (Deemed to be University)


