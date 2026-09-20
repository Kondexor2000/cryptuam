import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os


def resolve_docs_dir():
    if os.path.isdir("docs") and os.listdir("docs"):
        return "docs"
    if os.path.isdir("docs_not") and os.listdir("docs_not"):
        return "docs_not"
    return "docs"


# ----------------------------
# 1️⃣ Embedding model (TensorFlow)
# ----------------------------
url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(url)

# ----------------------------
# 2️⃣ Indeks + dokumenty
# ----------------------------

index_file = "docs.index"
npy_file = "docs.npy"

docs_dir = resolve_docs_dir()

index = faiss.read_index(index_file)
docs = np.load(npy_file, allow_pickle=True)

# ----------------------------
# 3️⃣ Model QA (Flan-T5 - TensorFlow)
# ----------------------------

g_flan = "google/flan-t5-base"

qa_tokenizer = AutoTokenizer.from_pretrained(g_flan)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(g_flan)

qa_generator = pipeline(
    "text-generation",
    model=qa_model,
    tokenizer=qa_tokenizer
)

# ----------------------------
# 4️⃣ Model do pomysłów (GPT2 PL - TensorFlow)
# ----------------------------
IDEA_MODEL = "radlab/polish-gpt2-small-v2"

idea_tokenizer = AutoTokenizer.from_pretrained(IDEA_MODEL)
idea_model = AutoModelForCausalLM.from_pretrained(IDEA_MODEL)

if idea_tokenizer.pad_token_id is None:
    idea_tokenizer.pad_token = idea_tokenizer.eos_token

idea_generator = pipeline(
    "text-generation",
    model=idea_model,
    tokenizer=idea_tokenizer,
)

# ----------------------------
# 5️⃣ banned words
# ----------------------------
try:
    banned_dir = "docs_not" if os.path.isdir("docs_not") else "docs"
    banned_words = []
    for file in os.listdir(banned_dir):
        with open(f"{banned_dir}/{file}", "r", encoding="utf-8") as f:
            banned_words.extend(line.strip().lower() for line in f.readlines() if line.strip())
except:
    banned_words = []

# ----------------------------
# 6️⃣ Chat loop
# ----------------------------

def generate_ideas(answer, language="pl", limit=3):
    if language == "pl":
        prompt = f"Temat: {answer}\nPodaj {limit} krótkie, merytoryczne pomysły po polsku:\n"
    else:
        prompt = f"Topic: {answer}\nGive {limit} short, meaningful ideas in English:\n"

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
        text = item["generated_text"].replace(prompt, "").strip()
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-*. 0123456789")
            if cleaned and cleaned not in ideas:
                ideas.append(cleaned)
            if len(ideas) >= limit:
                return ideas[:limit]
    return ideas[:limit]


def main():
    print("Mini ChatGPT (na Twoich dokumentach). Wpisz 'exit', aby zakończyć.\n")

    while True:
        question = input("Ty: ")

        if question.lower() in ["exit", "quit"]:
            break

        # 🔒 filtr
        if any(word in question.lower() for word in banned_words):
            print("AI: Nie mogę wygenerować tekstów naruszających zasady etyczne.")
            continue

        # 🔎 embedding (TensorFlow)
        q_embedding = embed_model([question]).numpy()
        faiss.normalize_L2(q_embedding)

        # 🔎 search
        D, I = index.search(np.array(q_embedding), k=5)
        context_many = "\n".join([docs[i] for i in I[0]])
        context_one = docs[I[0][0]]

        # 🧠 prompt QA
        prompt_many = f"""
Odpowiedz na pytanie na podstawie kontekstu.

Kontekst:
{context_many}

Pytanie:
{question}

Odpowiedź:
"""

        prompt_one = f"""
Odpowiedz na pytanie na podstawie kontekstu.

Kontekst:
{context_one}

Pytanie:
{question}

Odpowiedź:
"""

        result_many = qa_generator(prompt_many)
        answer_many = result_many[0]["generated_text"].strip()

        result_one = qa_generator(prompt_one)
        answer_one = result_one[0]["generated_text"].strip()

        ideas_many_pl = generate_ideas(answer_many, language="pl")
        ideas_many_en = generate_ideas(answer_many, language="en")
        ideas_one_pl = generate_ideas(answer_one, language="pl")
        ideas_one_en = generate_ideas(answer_one, language="en")

        # 📢 output
        print("\nAI_out:", answer_many)
        print("\nAI_in:", answer_one)

        if ideas_many_pl:
            print("\n💡 Pomysły_zewnętrzne (PL):")
            for i, idea in enumerate(ideas_many_pl, 1):
                print(f"{i}. {idea}")

        if ideas_many_en:
            print("\n💡 Pomysły_zewnętrzne (EN):")
            for i, idea in enumerate(ideas_many_en, 1):
                print(f"{i}. {idea}")

        if ideas_one_pl:
            print("\n💡 Pomysły_wewnętrzne (PL):")
            for i, idea in enumerate(ideas_one_pl, 1):
                print(f"{i}. {idea}")

        if ideas_one_en:
            print("\n💡 Pomysły_wewnętrzne (EN):")
            for i, idea in enumerate(ideas_one_en, 1):
                print(f"{i}. {idea}")

        print("\n" + "-"*50)


if __name__ == "__main__":
    main()
