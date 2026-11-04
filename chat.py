import os
import re
import json
import urllib.error
import urllib.request

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    import tensorflow_hub as hub
except ImportError:  # pragma: no cover
    hub = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None

def resolve_docs_dir():
    if os.path.isdir("docs") and os.listdir("docs"):
        return "docs"
    if os.path.isdir("docs_not") and os.listdir("docs_not"):
        return "docs_not"
    return "docs"


def load_text_files(directory):
    if not os.path.isdir(directory):
        return []

    docs = []
    for file in sorted(os.listdir(directory)):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(directory, file)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
            if text:
                docs.append(text)
        except OSError:
            continue
    return docs


def load_banned_words():
    banned_dir = "docs_not" if os.path.isdir("docs_not") else "docs"
    banned_words = []
    try:
        for file in sorted(os.listdir(banned_dir)):
            path = os.path.join(banned_dir, file)
            with open(path, "r", encoding="utf-8") as handle:
                banned_words.extend(
                    line.strip().lower() for line in handle if line.strip()
                )
    except OSError:
        return []
    return banned_words


def safe_embed_model(texts):
    if not hasattr(safe_embed_model, "backend"):
        safe_embed_model.backend = None

    if safe_embed_model.backend is None:
        return np.asarray([[] for _ in texts])

    if isinstance(texts, str):
        texts = [texts]

    return safe_embed_model.backend(texts)


def build_embedding_backend(docs):
    if hub is not None:
        try:
            url = "https://tfhub.dev/google/universal-sentence-encoder/4"
            embed_model = hub.load(url)

            def backend(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return np.asarray(embed_model(texts).numpy(), dtype=np.float32)

            safe_embed_model.backend = backend
            return "tfhub"
        except Exception:
            pass

    if TfidfVectorizer is not None and cosine_similarity is not None and docs:
        vectorizer = TfidfVectorizer()
        doc_matrix = vectorizer.fit_transform(docs)

        def backend(texts):
            if isinstance(texts, str):
                texts = [texts]
            return vectorizer.transform(texts).toarray()

        safe_embed_model.backend = backend
        return {"mode": "tfidf", "docs": docs, "matrix": doc_matrix, "vectorizer": vectorizer}

    def backend(texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray([[1.0] for _ in texts], dtype=np.float32)

    safe_embed_model.backend = backend
    return "simple"


# ----------------------------
# Dokumenty i banlisty
# ----------------------------
# Asystent korzysta z docs_speech i dodatkowego słownictwa z docs_listen.
# Zachowujemy dotychczasowy katalog dokumentów jako zgodność wsteczną.
listen_docs = load_text_files("docs_listen")
speech_docs = load_text_files("docs_speech")
docs_dir = "docs_speech" if speech_docs else resolve_docs_dir()
docs = speech_docs or load_text_files(docs_dir)

index = None
if docs and faiss is not None:
    try:
        embeddings = safe_embed_model(docs)
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.asarray(embeddings, dtype=np.float32))
        np.save("docs.npy", docs)
        faiss.write_index(index, "docs.index")
    except Exception:
        index = None

embedding_backend = build_embedding_backend(docs)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")

banned_words = load_banned_words()


def _find_best_context(question, top_k=5):
    if not docs:
        return "", []

    if isinstance(embedding_backend, dict) and embedding_backend.get("mode") == "tfidf":
        vectorizer = embedding_backend["vectorizer"]
        matrix = embedding_backend["matrix"]
        q_vector = vectorizer.transform([question])
        scores = cosine_similarity(q_vector, matrix).ravel()
        if not np.any(scores > 0):
            return "", []
        indices = [
            int(idx) for idx in np.argsort(scores)[::-1] if scores[idx] > 0
        ][:top_k]
        selected = [docs[i] for i in indices if 0 <= i < len(docs)]
        return "\n".join(selected), indices

    if index is not None:
        q_embedding = safe_embed_model(question)
        if q_embedding.ndim == 1:
            q_embedding = q_embedding.reshape(1, -1)

        if hasattr(faiss, "normalize_L2"):
            faiss.normalize_L2(q_embedding)
        D, I = index.search(np.asarray(q_embedding, dtype=np.float32), k=min(top_k, len(docs)))
        selected = [docs[i] for i in I[0] if 0 <= i < len(docs)]
        return "\n".join(selected), I[0].tolist()

    normalized_question = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", question.lower())
    ranked = []
    for idx, text in enumerate(docs):
        text_tokens = set(re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", text.lower()))
        score = sum(1 for token in normalized_question if token in text_tokens)
        ranked.append((score, idx, text))
    ranked.sort(key=lambda row: row[0], reverse=True)
    if not ranked or ranked[0][0] == 0:
        return "", []
    selected = [text for _, _, text in ranked[:top_k]]
    return "\n".join(selected), [idx for _, idx, _ in ranked[:top_k]]


def _find_listen_context(question, top_k=8):
    """Find vocabulary lines in docs_listen that relate to the user's question."""
    stop_words = {
        "a", "an", "and", "are", "co", "czy", "do", "for", "how", "i",
        "in", "is", "it", "jak", "jest", "na", "of", "or", "the", "to",
        "w", "what", "when", "where", "with", "z", "za", "ze",
    }
    question_tokens = [
        token for token in re.findall(
            r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", question.casefold()
        ) if token not in stop_words
    ]
    if not question_tokens:
        return "", []

    question_terms = set(question_tokens)
    normalized_question = " ".join(question_tokens)
    ranked = []
    for document_index, document in enumerate(listen_docs):
        for line_index, line in enumerate(document.splitlines()):
            line = line.strip()
            if not line:
                continue
            normalized_line = re.findall(
                r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", line.casefold()
            )
            overlap = len(question_terms & set(normalized_line))
            if overlap:
                phrase_match = int(normalized_question in " ".join(normalized_line))
                ranked.append((phrase_match, overlap, -document_index, -line_index, line, document_index))

    ranked.sort(reverse=True)
    selected = ranked[:top_k]
    return "\n".join(item[4] for item in selected), [item[5] for item in selected]


def _fallback_answer(question, context):
    text = context.strip()
    if not text:
        return "Nie znalazłem odpowiedniego kontekstu w dokumentach."

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    question_terms = set(re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", question.lower()))

    if not question_terms:
        return lines[0][:500]

    best_line = None
    best_score = -1
    for line in lines:
        line_tokens = set(re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", line.lower()))
        score = sum(1 for token in question_terms if token in line_tokens)
        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None:
        return lines[0][:500]
    if best_score <= 0:
        return "Nie znalazłem odpowiedniego kontekstu w dokumentach."

    return best_line[:500]


def _chat_answer(question, context, conversation_history=None):
    """Generate a natural Polish reply using local vocabulary lists as context."""
    messages = [{
        "role": "system",
        "content": (
            "Jesteś życzliwym, naturalnym asystentem rozmawiającym po polsku. "
            "Odpowiadaj na to, co użytkownik rzeczywiście napisał; nie traktuj "
            "każdej wiadomości jak testu słownictwa. Gdy pyta o angielskie słowo, "
            "wyjaśnij jego znaczenie po polsku i podaj krótki przykład. "
            "Listy poniżej są materiałami pomocniczymi: użyj ich, gdy pasują do rozmowy, "
            "ale nie twierdź, że zawierają definicję, jeśli jej tam nie ma. "
            "Jeśli nie znasz odpowiedzi, powiedz to wprost. Odpowiadaj zwięźle."
        ),
    }]
    for user_text, assistant_text in (conversation_history or [])[-4:]:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    prompt = (
        "Materiały z docs_speech i docs_listen:\n"
        f"{context[:7000] or '(brak dopasowanych haseł)'}\n\n"
        f"Wiadomość użytkownika: {question}"
    )
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.5, "num_predict": 180},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = result.get("message", {}).get("content", "").strip()
        return answer or "Nie udało mi się ułożyć odpowiedzi. Spróbuj napisać to inaczej."
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return f"Nie widzę modelu {OLLAMA_MODEL}. Pobierz go poleceniem: ollama pull {OLLAMA_MODEL}"
        return f"Ollama zwróciła błąd ({error.code}). Sprawdź, czy działa usługa Ollama."
    except (urllib.error.URLError, TimeoutError):
        return (
            "Nie mogę połączyć się z Ollama. Zainstaluj Ollama, uruchom ją i pobierz model: "
            f"ollama pull {OLLAMA_MODEL}"
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return "Ollama zwróciła nieprawidłową odpowiedź. Spróbuj ponownie."


def main():
    print("Rozmowa z asystentem. Napisz, co masz do powiedzenia; wpisz 'exit', aby zakończyć.\n")
    conversation_history = []

    while True:
        question = input("Ty: ").strip()

        if question.strip().lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        if any(word in question.lower() for word in banned_words):
            response = "Nie mogę odpowiedzieć na tę wiadomość. Możesz powiedzieć coś innego?"
            print(f"\nAsystent: {response}\n")
            conversation_history.append((question, response))
            continue

        if not docs and not listen_docs:
            response = "Nie mam teraz materiałów, na których mógłbym oprzeć odpowiedź."
            print(f"\nAsystent: {response}\n")
            conversation_history.append((question, response))
            continue

        listen_context, _ = _find_listen_context(question)
        speech_context, _ = _find_best_context(question, top_k=5)
        context = "\n".join(part for part in (
            f"docs_speech:\n{speech_context}" if speech_context else "",
            f"docs_listen:\n{listen_context}" if listen_context else "",
        ) if part)
        response = _chat_answer(question, context, conversation_history[-4:])
        print(f"\nAsystent: {response}\n")
        conversation_history.append((question, response))


if __name__ == "__main__":
    main()
