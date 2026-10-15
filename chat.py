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
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
except ImportError:  # pragma: no cover
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForCausalLM = None


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
docs_dir = resolve_docs_dir()
docs = load_text_files(docs_dir)

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
qa_generator = None
if pipeline is not None and AutoTokenizer is not None and AutoModelForSeq2SeqLM is not None:
    try:
        g_flan = "google/flan-t5-base"
        qa_tokenizer = AutoTokenizer.from_pretrained(g_flan)
        qa_model = AutoModelForSeq2SeqLM.from_pretrained(g_flan)
        qa_generator = pipeline("text-generation", model=qa_model, tokenizer=qa_tokenizer)
    except Exception:
        qa_generator = None

# ----------------------------
# Model do pomysłów
# ----------------------------
idea_generator = None
if pipeline is not None and AutoTokenizer is not None and AutoModelForCausalLM is not None:
    try:
        idea_model_name = "radlab/polish-gpt2-small-v2"
        idea_tokenizer = AutoTokenizer.from_pretrained(idea_model_name)
        idea_model = AutoModelForCausalLM.from_pretrained(idea_model_name)

        if getattr(idea_tokenizer, "pad_token_id", None) is None:
            idea_tokenizer.pad_token = idea_tokenizer.eos_token

        idea_generator = pipeline(
            "text-generation",
            model=idea_model,
            tokenizer=idea_tokenizer,
        )
    except Exception:
        idea_generator = None

banned_words = load_banned_words()


def _find_best_context(question, top_k=5):
    if not docs:
        return "", []

    if isinstance(embedding_backend, dict) and embedding_backend.get("mode") == "tfidf":
        vectorizer = embedding_backend["vectorizer"]
        matrix = embedding_backend["matrix"]
        q_vector = vectorizer.transform([question])
        scores = cosine_similarity(q_vector, matrix).ravel()
        indices = np.argsort(scores)[::-1][:top_k]
        selected = [docs[i] for i in indices if 0 <= i < len(docs)]
        return "\n".join(selected), indices.tolist()

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
    selected = [text for _, _, text in ranked[:top_k]]
    return "\n".join(selected), [idx for _, idx, _ in ranked[:top_k]]


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

    return best_line[:500]


def generate_ideas(answer, language="pl", limit=3, mode=0):
    if not answer:
        return []

    prompt_noise = {
        "temat", "topic", "podaj", "give", "krótkie", "short", "merytoryczne",
        "meaningful", "pomysły", "ideas", "idea", "odpowiedz", "answer",
        "pytanie", "question", "podstawie", "based", "context", "kontekst",
        "na", "the", "in", "on"
    }

    mode = max(0, min(3, int(mode)))
    mode_prompts = {
        0: "brief, direct, and practical",
        1: "clear, polished, and user-focused",
        2: "creative, bold, and original",
        3: "analytical, strategic, and actionable",
    }

    if idea_generator is not None:
        try:
            if language == "pl":
                prompt = f"Temat: {answer}\nPodaj {limit} krótkie, merytoryczne pomysły po polsku:\n"
            else:
                style = mode_prompts.get(mode, mode_prompts[0])
                prompt = (
                    f"Topic: {answer}\n"
                    f"Generate {limit} short, meaningful ideas in English in a {style} style.\n"
                )

            outputs = idea_generator(
                prompt,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                num_return_sequences=3,
            )

            ideas = []
            for item in outputs:
                text = item["generated_text"]
                if text.startswith(prompt):
                    text = text[len(prompt):]
                for line in text.splitlines():
                    cleaned = line.strip().lstrip("-*. 0123456789").strip()
                    if not cleaned:
                        continue
                    norm = cleaned.lower()
                    root = norm.split()[0].strip(",:;.-") if norm.split() else ""
                    if norm in prompt_noise or root in prompt_noise:
                        continue
                    if any(token in prompt_noise for token in norm.split()):
                        if len(cleaned.split()) <= 2:
                            continue
                    if cleaned not in ideas:
                        ideas.append(cleaned)
                    if len(ideas) >= limit:
                        return ideas[:limit]
            if ideas:
                return ideas[:limit]
        except Exception:
            pass

    keywords = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]{4,}", answer)
    seen = set()
    ideas = []
    for item in keywords:
        norm = item.lower()
        if norm not in seen:
            seen.add(norm)
            ideas.append(item)
        if len(ideas) >= limit:
            break

    if not ideas:
        ideas = [answer.strip()[:80]]

    if language == "pl":
        return ideas[:limit]
    return [f"{idea} in context" for idea in ideas[:limit]]


def extract_keywords(text, limit=3):
    """Extract distinct, non-stopword terms from an idea."""
    if limit <= 0:
        return []

    stop_words = {
        "a", "aby", "albo", "ale", "aż", "bo", "by", "być", "co", "czy",
        "dla", "do", "gdzie", "i", "ich", "im", "in", "jest", "jak", "jako",
        "je", "jego", "jej", "już", "lub", "ma", "mi", "mnie", "na", "nad",
        "nie", "oraz", "o", "od", "of", "on", "or", "po", "pod", "przez",
        "przy", "się", "są", "ta", "tak", "te", "ten", "to", "the", "this",
        "that", "they", "their", "then", "there", "these", "those", "we", "w",
        "with", "you", "your", "za", "ze", "z", "and", "are", "as", "at",
        "be", "but", "for", "from", "is", "context",
    }
    keywords = []
    seen = set()
    for word in re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]{3,}", text):
        normalized = word.casefold()
        if normalized in stop_words or normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords


def _print_ideas(ideas, language="pl", mode=None):
    keyword_label = "Słowa kluczowe" if language == "pl" else "Keywords"
    for i, idea in enumerate(ideas, 1):
        mode_label = f"[Mode {mode}] " if mode is not None else ""
        print(f"{i}. {mode_label}{idea}")
        keywords = extract_keywords(idea)
        if keywords:
            print(f"   {keyword_label}: {', '.join(keywords)}")


def main():
    print("Mini ChatGPT (na Twoich dokumentach). Wpisz 'exit', aby zakończyć.\n")

    while True:
        question = input("Ty: ")

        if question.lower() in ["exit", "quit"]:
            break

        if any(word in question.lower() for word in banned_words):
            print("AI: Nie mogę wygenerować tekstów naruszających zasady etyczne.")
            continue

        if not docs:
            print("AI: Brak dokumentów do przeszukania.")
            continue

        context_many, _ = _find_best_context(question, top_k=5)
        context_one = docs[0]
        if context_many:
            context_one = context_many.splitlines()[0]

        if qa_generator is not None:
            prompt_many = f"""
Odpowiedz na pytanie na podstawie kontekstu.

Kontekst:
{context_many}

Pytanie:
{question}

Odpowiedź:
"""
            result_many = qa_generator(prompt_many)
            answer_many = result_many[0]["generated_text"].strip()

            prompt_one = f"""
Odpowiedz na pytanie na podstawie kontekstu.

Kontekst:
{context_one}

Pytanie:
{question}

Odpowiedź:
"""
            result_one = qa_generator(prompt_one)
            answer_one = result_one[0]["generated_text"].strip()
        else:
            answer_many = _fallback_answer(question, context_many)
            answer_one = _fallback_answer(question, context_one)

        ideas_many_pl = generate_ideas(answer_many, language="pl")
        ideas_many_en = [
            generate_ideas(answer_many, language="en", mode=mode)
            for mode in range(4)
        ]
        ideas_one_pl = generate_ideas(answer_one, language="pl")
        ideas_one_en = [
            generate_ideas(answer_one, language="en", mode=mode)
            for mode in range(4)
        ]

        print("\nAI_out:", answer_many)
        print("\nAI_in:", answer_one)

        if ideas_many_pl:
            print("\n💡 Pomysły_zewnętrzne (PL):")
            _print_ideas(ideas_many_pl)

        for mode, ideas in enumerate(ideas_many_en):
            if ideas:
                print(f"\n💡 Pomysły_zewnętrzne (EN, Mode {mode}):")
                _print_ideas(ideas, language="en", mode=mode)

        if ideas_one_pl:
            print("\n💡 Pomysły_wewnętrzne (PL):")
            _print_ideas(ideas_one_pl)

        for mode, ideas in enumerate(ideas_one_en):
            if ideas:
                print(f"\n💡 Pomysły_wewnętrzne (EN, Mode {mode}):")
                _print_ideas(ideas, language="en", mode=mode)

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
