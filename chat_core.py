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
RETRY_MESSAGE = "Please try again using at least one word from the docs_speech materials."


def _has_speech_word(question, documents):
    question_words = {token.casefold() for token in TOKEN_RE.findall(question)}
    speech_words = {
        token.casefold()
        for document in documents
        for token in TOKEN_RE.findall(document["content"])
    }
    return bool(question_words & speech_words)


def _listen_reply(question, speech_answer, listen_documents):
    """Reply in English with a phrase from docs_listen related to the match."""
    listen_lines = [
        line.strip()
        for document in listen_documents
        for line in document["content"].splitlines()
        if line.strip()
    ]
    if not listen_lines:
        return f"I found “{speech_answer}” in docs_speech, but there are no docs_listen phrases to connect it to."

    reference_words = {
        token.casefold()
        for token in TOKEN_RE.findall(f"{question} {speech_answer}")
    }
    related_line = next(
        (line for line in listen_lines
         if reference_words & {token.casefold() for token in TOKEN_RE.findall(line)}),
        listen_lines[0],
    )
    return f'A related listening phrase is “{related_line}”. Try using it in an English sentence.'


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


def _ideas_for_context(text, language="pl", limit=3):
    ideas = ideas_from_answer(text, limit)
    if not ideas and text.strip():
        ideas = [text.strip()[:80]]
    if language == "en":
        return [f"{idea} in context" for idea in ideas]
    return ideas


def _idea_group(label, external, internal):
    return {
        "label": label,
        "external_pl": _ideas_for_context(external, "pl"),
        "external_en": _ideas_for_context(external, "en"),
        "internal_pl": _ideas_for_context(internal, "pl"),
        "internal_en": _ideas_for_context(internal, "en"),
    }


def answer_question(question, documents, listen_documents=None):
    """Format terminal-style retrieval answers and prototype ideas for the browser."""
    listen_documents = listen_documents or []
    if not _has_speech_word(question, documents):
        return {
            "answer_out": RETRY_MESSAGE,
            "answer_in": "",
            "source": "brak dopasowania",
            "idea_groups": [],
            "mode": "retry",
        }

    match = _best_match(question, documents)
    if not match:
        return {
            "answer_out": NO_MATCH,
            "answer_in": NO_MATCH,
            "source": "brak dopasowania",
            "idea_groups": [],
            "mode": "retrieval",
        }

    speech_answer = match["answer"]
    answer_out = _listen_reply(question, speech_answer, listen_documents)
    source = match["source"]

    selected_document = next(
        (document for document in documents if document["file"] == source),
        None,
    )
    first_line = next(
        (line.strip() for line in (selected_document or {}).get("content", "").splitlines() if line.strip()),
        answer_out,
    )
    answer_in = first_line[:500]
    idea_groups = [_idea_group("biblioteka", answer_out, answer_in)]

    if listen_documents:
        listen_text = "\n".join(document["content"] for document in listen_documents)
        external_context = (
            "Materiał słownictwa (docs_listen):\n"
            + listen_text
            + f"\n\nTekst użytkownika:\n{question}"
            + f"\n\nKontekst odpowiedzi z docs_speech:\n{answer_out}"
        )
        internal_context = (
            "Materiał słownictwa (docs_listen):\n"
            + listen_text
            + f"\n\nTekst użytkownika:\n{question}"
            + f"\n\nKontekst odpowiedzi z docs_speech:\n{answer_in}"
        )
        idea_groups.append(
            _idea_group(
                "współpraca z docs_listen i tekstem użytkownika",
                external_context,
                internal_context,
            )
        )

    return {
        "answer_out": answer_out,
        "answer_in": answer_in,
        "source": source,
        "idea_groups": idea_groups,
        "mode": "retrieval",
    }
