from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import List

from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    answer: str
    context: str
    scores: List[float]


class PDFRAG:
    """A lightweight RAG pipeline for PDF question answering.

    Design goals:
    - Retrieve context only from uploaded PDF
    - Never hallucinate outside retrieved context
    - Return "I don't know" when evidence is weak
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 70,
        min_score: float = 0.07,
        model_name: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_score = min_score
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()

        self.chunks: List[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.chunk_vectors = None
        self._client: Groq | None = None

    def ingest_pdf(self, pdf_bytes: bytes) -> int:
        return self.ingest_pdfs([pdf_bytes])

    def ingest_pdfs(self, pdf_bytes_list: List[bytes]) -> int:
        all_text_parts: List[str] = []

        for pdf_bytes in pdf_bytes_list:
            text = self._extract_text(pdf_bytes)
            if text.strip():
                all_text_parts.append(text)

        combined_text = "\n\n".join(all_text_parts)
        self.chunks = self._chunk_text(combined_text)

        if not self.chunks:
            raise ValueError("No readable text found in the uploaded PDF files.")

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        return len(self.chunks)

    def ask(self, question: str, top_k: int = 5) -> RetrievalResult:
        if not self.vectorizer or self.chunk_vectors is None or not self.chunks:
            raise ValueError("No document indexed. Upload and process PDF files first.")

        q = question.strip()
        if not q:
            return RetrievalResult(answer="I don't know.", context="", scores=[])

        query_vector = self.vectorizer.transform([q])
        sim = cosine_similarity(query_vector, self.chunk_vectors)[0]

        top_k = min(top_k, len(self.chunks))
        top_ids = sim.argsort()[::-1][:top_k]
        top_scores = [float(sim[i]) for i in top_ids]

        if not top_scores or top_scores[0] < self.min_score:
            return RetrievalResult(answer="I don't know.", context="", scores=top_scores)

        retrieved_chunks = [self.chunks[i] for i in top_ids]
        context = "\n\n".join(retrieved_chunks)
        answer = self._answer_with_groq(question=q, context=context)

        if not answer:
            answer = "I don't know."

        return RetrievalResult(answer=answer, context=context, scores=top_scores)

    def _extract_text(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text.strip())

        return "\n".join([p for p in pages if p])

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        for start in range(0, len(words), step):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            if not chunk_words:
                continue
            chunks.append(" ".join(chunk_words))

        return chunks

    def _answer_with_groq(self, question: str, context: str) -> str:
        if not self.api_key:
            return "I don't know."

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                max_tokens=256,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You answer questions using only the provided context from a PDF transcript. "
                            "If the answer is not explicitly supported by the context, reply exactly: I don't know. "
                            "Do not use outside knowledge. Keep the answer concise. "
                            "When you answer, include a short exact quote from the context in parentheses."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\n"
                            f"Question: {question}\n\n"
                            "Answer using only the context above."
                        ),
                    },
                ],
            ) 

            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                return "I don't know."
            if answer.lower().startswith("i don't know"):
                return "I don't know."
            if "(" in answer and ")" in answer:
                return answer
            return answer
        except Exception:
            return "I don't know."

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _keywords(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        return {t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2}

