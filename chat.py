import os
import re

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

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:  # pragma: no cover
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None


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

# ----------------------------
# Model QA (Flan-T5)
# ----------------------------
qa_tokenizer = None
qa_model = None
if AutoTokenizer is not None and AutoModelForSeq2SeqLM is not None:
    try:
        g_flan = "google/flan-t5-base"
        qa_tokenizer = AutoTokenizer.from_pretrained(g_flan)
        qa_model = AutoModelForSeq2SeqLM.from_pretrained(g_flan)
        qa_model.eval()
    except Exception:
        qa_tokenizer = None
        qa_model = None

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
    """Generate one short, grounded reply; fall back to a matching source line."""
    if not context.strip():
        return _fallback_answer(question, context)
    if qa_model is None or qa_tokenizer is None:
        return _fallback_answer(question, context)

    history_text = "\n".join(
        f"User: {user_text}\nAssistant: {assistant_text}"
        for user_text, assistant_text in (conversation_history or [])[-4:]
    )
    history_section = f"\nConversation so far:\n{history_text}\n" if history_text else ""
    prompt = (
        "Answer the user's question briefly and naturally in the same language. "
        "Use only the context below. If the context does not contain the answer, "
        "say that clearly. Do not invent examples or repeat the prompt.\n\n"
        f"Context from docs_listen:\n{context[:5000]}\n"
        f"{history_section}\nCurrent question: {question}\nAnswer:"
    )
    try:
        import torch

        inputs = qa_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.inference_mode():
            output = qa_model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=4,
                early_stopping=True,
            )
        answer = qa_tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if answer:
            return answer
    except Exception:
        pass
    return _fallback_answer(question, context)


def main():
    print("Chatbot materiałów językowych. Pytaj o hasła z dokumentów; wpisz 'exit', aby zakończyć.\n")
    conversation_history = []
    previous_speech_match = ""

    while True:
        question = input("Ty: ")

        if question.strip().lower() in ["exit", "quit"]:
            break

        if not question.strip():
            continue

        if any(word in question.lower() for word in banned_words):
            print("\nAsystent: Nie mogę pomóc w tym zapytaniu. Spróbuj zapytać o inne hasło.\n")
            continue

        if not docs and not listen_docs:
            print("\nAsystent: Nie ma dostępnych dokumentów do przeszukania.\n")
            continue

        speech_query = question
        if previous_speech_match:
            speech_query = f"{question}\n{previous_speech_match}"
        speech_context, _ = _find_best_context(speech_query, top_k=5)
        speech_match = _fallback_answer(speech_query, speech_context)
        no_match = _fallback_answer(question, "")

        print("\nAsystent:")
        if speech_match == no_match:
            response = "Nie znalazłem tego hasła w docs_speech. Możesz podać krótszą frazę albo zdanie, w którym go używasz?"
            print(f"  {response}")
            conversation_history.append((question, response))
            print()
            continue

        print(f"  Dopasowanie wejścia (docs_speech): {speech_match}")
        previous_context = previous_speech_match if previous_speech_match != speech_match else ""
        listen_query = f"{question}\n{speech_match}\n{previous_context}"
        listen_context, _ = _find_listen_context(listen_query)
        if not listen_context:
            response = (
                f"Rozumiem, pytasz o „{speech_match}”. Nie widzę tego hasła w docs_listen, "
                "więc nie chcę zgadywać. W jakim zdaniu albo znaczeniu mam Ci pomóc?"
            )
            print(f"  {response}")
            conversation_history.append((question, response))
            previous_speech_match = speech_match
            print()
            continue

        history = conversation_history[-4:]
        answer = _chat_answer(question, listen_context, history) if history else _chat_answer(question, listen_context)
        if answer == no_match:
            response = (
                f"Dopasowałem „{speech_match}” w docs_speech, ale nie znalazłem odpowiedzi "
                "w docs_listen. Możesz doprecyzować, o co chcesz zapytać?"
            )
            print(f"  {response}")
        else:
            response = answer
            print(f"\n  {response}")
            print("\n  Źródło odpowiedzi: docs_listen.")
            print("  Dopasowanie pytania pochodzi z docs_speech.")
        conversation_history.append((question, response))
        previous_speech_match = speech_match
        print()


if __name__ == "__main__":
    main()
