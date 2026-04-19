# PDF Assistant (Simple RAG)

A simple PDF question-answering app built with a lightweight RAG pipeline and Groq for generation.

## What this app does

- Takes one or more PDFs as input.
- Indexes all uploaded PDF text into chunks.
- Retrieves the most relevant chunks for each user question.
- Answers **only** from retrieved document context using Groq.
- If context is not sufficient, returns: **"I don't know."**

## Steps Performed To Create This RAG Model

1. Created project files:
   - `app.py` for UI.
   - `rag_engine.py` for the RAG logic.
   - `requirements.txt` for dependencies.
2. Implemented PDF text extraction in `rag_engine.py` using `pypdf`.
3. Added support for uploading and indexing multiple PDF files together.
4. Added text chunking with overlap to preserve context across boundaries.
5. Built a retriever with `TfidfVectorizer` and cosine similarity (`scikit-learn`).
6. Implemented top-k chunk retrieval for each question.
7. Added strict fallback behavior:
   - If best similarity score is below threshold, answer is "I don't know."
   - If Groq cannot produce a grounded answer, answer is "I don't know."
8. Added Groq-based answer generation with a prompt that only allows context-grounded responses and asks for a short exact quote from the source context.
9. Built a clean Streamlit UI with:
   - PDF upload and indexing button.
   - Question input and answer display.
   - Expanders for retrieved context and similarity scores.
10. Added guardrails in UI and backend for missing input / missing indexed document.
11. Added `.env` support for `GROQ_API_KEY` and optional `GROQ_MODEL`.

## Project Structure

```
pdf-assistant/
  app.py
  rag_engine.py
   requirements.txt
   .env.example
  README.md
```

## Setup

1. Create and activate a virtual environment.
2. Create a `.env` file and add your Groq key:

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

5. Open the URL shown in terminal (usually `http://localhost:8501`).

## Notes

- This version uses local retrieval plus Groq for answer generation.
- The answer is restricted by prompt and retrieval thresholds, with "I don't know." as fallback.
- For scanned/image-only PDFs, text extraction may fail unless OCR is added.
