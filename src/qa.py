"""RAG question answering for DocuMind.

Two answer modes:

* **extractive** (default, fully offline): the most relevant retrieved
  passages are returned as the answer, each with its page citation. Nothing
  is generated, so nothing can be hallucinated — every claim is a quote
  from the document.
* **generative** (optional): if the ``OPENAI_API_KEY`` environment variable
  is set, gpt-4o-mini writes a short answer strictly from the retrieved
  context, with page citations. If the key is absent, DocuMind falls back
  to extractive mode automatically and says so.

Both modes refuse to answer from outside knowledge: when the retrieved
chunks do not contain the answer, they say the document does not cover it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .embeddings import Embedder
from .ingest import Chunk
from .vectorstore import VectorStore

DEFAULT_TOP_K = 5
GENERATIVE_MODEL = "gpt-4o-mini"
MIN_SCORE = 0.15  # below this, a chunk is too unrelated to cite


@dataclass
class Source:
    """One retrieved passage, with its page citation."""

    text: str
    page: int
    score: float

    def citation(self) -> str:
        return f"[p. {self.page}]"


@dataclass
class Answer:
    """The result of asking one question."""

    answer: str
    sources: list[Source]
    mode: str  # "extractive" or "generative"
    note: str = ""


def _relevant(results: list[tuple[Chunk, float]]) -> list[Source]:
    """Keep only chunks similar enough to be worth citing."""
    return [
        Source(text=chunk.text, page=chunk.page, score=score)
        for chunk, score in results
        if score >= MIN_SCORE
    ]


def _extractive_answer(question: str, sources: list[Source]) -> str:
    if not sources:
        return (
            "The document does not seem to cover this. "
            "Try rephrasing the question or asking about a topic in the document."
        )
    lines = [
        "Based on the document, here are the most relevant passages:",
        "",
    ]
    for i, source in enumerate(sources, start=1):
        lines.append(f"{i}. {source.text} {source.citation()}")
    return "\n".join(lines)


def _generative_answer(question: str, sources: list[Source]) -> str:
    from openai import OpenAI

    context = "\n\n".join(
        f"[Page {s.page}] {s.text}" for s in sources
    )
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=GENERATIVE_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using ONLY the provided document "
                    "passages. Answer concisely in 2-4 sentences. After each "
                    "claim, add the page citation exactly as given, e.g. "
                    "[p. 2]. If the passages do not contain the answer, say "
                    "the document does not cover it — do not use outside "
                    "knowledge and do not guess."
                ),
            },
            {
                "role": "user",
                "content": f"Document passages:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content.strip()


class DocuMindQA:
    """A RAG assistant bound to one indexed document."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def ask(self, question: str, top_k: int = DEFAULT_TOP_K) -> Answer:
        """Answer ``question`` from the indexed document."""
        query_embedding = self.embedder.embed_one(question)
        results = self.store.search(query_embedding, top_k=top_k)
        sources = _relevant(results)

        if os.environ.get("OPENAI_API_KEY"):
            mode = "generative"
            note = "Answered with gpt-4o-mini using only the retrieved passages."
            answer_text = (
                _generative_answer(question, sources)
                if sources
                else _extractive_answer(question, sources)
            )
        else:
            mode = "extractive"
            note = (
                "Extractive mode (no OPENAI_API_KEY set): the answer is the "
                "most relevant passages from the document itself."
            )
            answer_text = _extractive_answer(question, sources)

        return Answer(answer=answer_text, sources=sources, mode=mode, note=note)
