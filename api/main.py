"""Small, database-free API that exposes the repository's Python tools."""

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from research_lab import build_brief


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+")
STOP_WORDS = {
    "aby", "albo", "ale", "and", "are", "as", "at", "be", "but", "by",
    "czy", "dla", "do", "from", "i", "in", "is", "it", "jak", "jest",
    "lub", "ma", "na", "nad", "nie", "of", "on", "or", "oraz", "że",
    "się", "the", "to", "w", "with", "z", "za", "ze",
}

app = FastAPI(
    title="AI Research Lab API",
    description="Database-free API for the research brief and document chatbot.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kondexor2000.github.io",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


def _load_speech_documents():
    documents = []
    directory = ROOT / "docs_speech"
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(text)
    return documents


def _answer_from_documents(question, documents):
    if not documents:
        return "Brak dokumentów do przeszukania."

    question_terms = set(TOKEN_RE.findall(question.casefold()))
    ranked = []
    for index, document in enumerate(documents):
        tokens = set(TOKEN_RE.findall(document.casefold()))
        ranked.append((len(question_terms & tokens), index, document))
    ranked.sort(key=lambda row: (-row[0], row[1]))

    selected = [document for _, _, document in ranked[:5]]
    lines = [line.strip() for document in selected for line in document.splitlines() if line.strip()]
    if not lines:
        return "Nie znalazłem odpowiedniego kontekstu w dokumentach."
    if not question_terms:
        return lines[0][:500]

    best_line = max(
        lines,
        key=lambda line: len(question_terms & set(TOKEN_RE.findall(line.casefold()))),
    )
    return best_line[:500]


def _ideas_from_answer(answer):
    ideas = []
    seen = set()
    for token in TOKEN_RE.findall(answer):
        normalized = token.casefold()
        if len(token) < 4 or normalized in STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        ideas.append(token)
        if len(ideas) == 3:
            break
    return ideas


@app.get("/api/health")
def health():
    return {"status": "ok", "database": False}


@app.get("/api/brief")
def research_brief():
    try:
        return build_brief(
            [ROOT / "docs_listen", ROOT / "docs_speech"],
            ROOT / "docs_not",
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=500, detail=f"Nie udało się zbudować briefu: {error}") from error


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Wpisz pytanie.")

    documents = _load_speech_documents()
    if not documents:
        raise HTTPException(status_code=503, detail="Brak materiałów w docs_speech.")

    answer = _answer_from_documents(question, documents)
    ideas = _ideas_from_answer(answer)
    return {
        "answer": answer,
        "ideas": ideas,
        "ideas_en": [f"{idea} in context" for idea in ideas],
        "source": "docs_speech",
        "mode": "retrieval",
    }
