"""Lightweight retrieval logic shared by the CLI, API, and browser runtime."""

import re
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+")
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "co", "could", "czy", "do", "does", "for", "from", "how", "i", "in", "is", "it", "jak", "jest",
    "na", "nad", "nie", "of", "o", "or", "oraz", "the", "to", "w", "with",
    "me", "meaning", "mean", "my", "please", "tell", "what", "which", "you", "your",
    "z", "za", "ze", "że", "się", "mnie", "mi", "powiedz", "wyjaśnij", "oznacza",
    "znaczy", "proszę",
}
NO_MATCH = "Nie znalazłem pasującego hasła w docs_speech. Spróbuj zapytać o konkretne słowo lub frazę z list."


def load_speech_documents(directory):
    """Load source names and non-empty .txt documents in stable order."""
    directory = Path(directory)
    return [
        {"file": path.name, "content": content}
        for path in sorted(directory.glob("*.txt"))
        if (content := path.read_text(encoding="utf-8").strip())
    ]


def _best_match(question, documents):
    if not documents:
        return None

    question_tokens = [
        token for token in TOKEN_RE.findall(question.casefold())
        if token not in STOP_WORDS
    ]
    normalized_question = " ".join(question_tokens)
    question_terms = set(question_tokens)
    if not question_terms:
        return None

    candidates = []
    for document_index, document in enumerate(documents):
        for line_index, line in enumerate(document["content"].splitlines()):
            line = line.strip()
            if not line:
                continue
            normalized_line = " ".join(TOKEN_RE.findall(line.casefold()))
            line_terms = set(normalized_line.split())
            overlap = len(question_terms & line_terms)
            minimum_overlap = 1 if len(question_terms) <= 2 else 2
            if overlap >= minimum_overlap:
                phrase_match = int(normalized_question in normalized_line)
                candidates.append((phrase_match, overlap, -document_index, -line_index, line, document["file"]))

    if not candidates:
        return None
    phrase_match, overlap, _, _, line, filename = max(candidates)
    return {"answer": line[:500], "source": filename, "matched_terms": overlap}


def answer_from_documents(question, documents):
    """Return only a source line with at least one meaningful term match."""
    match = _best_match(question, documents)
    return match["answer"] if match else NO_MATCH


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
    """Format a grounded lookup result for use by the browser."""
    match = _best_match(question, documents)
    return {
        "answer": match["answer"] if match else NO_MATCH,
        "source": match["source"] if match else "brak dopasowania",
        "mode": "retrieval",
    }
