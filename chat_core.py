"""Lightweight retrieval logic shared by the CLI, API, and browser runtime."""

import re
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+")
def load_speech_documents(directory):
    """Load non-empty .txt documents in stable filename order."""
    directory = Path(directory)
    return [
        content
        for path in sorted(directory.glob("*.txt"))
        if (content := path.read_text(encoding="utf-8").strip())
    ]


def answer_from_documents(question, documents):
    """Return the most relevant source line using the chatbot's lexical fallback."""
    if not documents:
        return "Brak dokumentów do przeszukania."

    question_terms = set(TOKEN_RE.findall(question.casefold()))
    ranked = []
    for index, document in enumerate(documents):
        tokens = set(TOKEN_RE.findall(document.casefold()))
        ranked.append((len(question_terms & tokens), index, document))
    ranked.sort(key=lambda row: (-row[0], row[1]))

    lines = [
        line.strip()
        for _, _, document in ranked[:5]
        for line in document.splitlines()
        if line.strip()
    ]
    if not lines:
        return "Nie znalazłem odpowiedniego kontekstu w dokumentach."
    if not question_terms:
        return lines[0][:500]

    best_line = max(
        lines,
        key=lambda line: len(question_terms & set(TOKEN_RE.findall(line.casefold()))),
    )
    return best_line[:500]


def ideas_from_answer(answer, limit=3):
    """Extract distinct keywords from an answer for the prototype's idea list."""
    ideas = []
    seen = set()
    for token in TOKEN_RE.findall(answer):
        normalized = token.casefold()
        if len(token) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        ideas.append(token)
        if len(ideas) >= limit:
            break
    return ideas


def answer_question(question, documents):
    """Format a chatbot response for use by the CLI/API/browser."""
    answer = answer_from_documents(question, documents)
    ideas = ideas_from_answer(answer)
    return {
        "answer": answer,
        "ideas": ideas,
        "ideas_en": [f"{idea} in context" for idea in ideas],
        "source": "docs_speech",
        "mode": "retrieval",
    }
