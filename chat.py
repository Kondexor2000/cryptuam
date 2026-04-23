import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

# ----------------------------
# 1️⃣ Embedding model
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 2️⃣ Indeks + dokumenty
# ----------------------------
index = faiss.read_index("docs.index")
docs = np.load("docs.npy", allow_pickle=True)

# ----------------------------
# 3️⃣ Model QA (Flan-T5)
# ----------------------------
qa_generator = pipeline(
    "text-generation",  
    model="google/flan-t5-base",
    max_new_tokens=150
)

# ----------------------------
# 4️⃣ Model do pomysłów (GPT2 PL)
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
    device=0 if torch.cuda.is_available() else -1
)

# ----------------------------
# 5️⃣ banned words (raz!)
# ----------------------------
try:
    for file in os.listdir("docs_not"):
        with open(f"docs_not/{file}", "r", encoding="utf-8") as f:
            banned_words = [line.strip().lower() for line in f.readlines()]
except:
    banned_words = []

# ----------------------------
# 6️⃣ Chat loop
# ----------------------------
print("Mini ChatGPT (na Twoich dokumentach). Wpisz 'exit', aby zakończyć.\n")

while True:
    question = input("Ty: ")

    if question.lower() in ["exit", "quit"]:
        break

    # 🔒 filtr
    if any(word in question.lower() for word in banned_words):
        print("AI: Nie mogę wygenerować tekstów naruszających zasady etyczne.")
        continue

    # 🔎 embedding
    q_embedding = embed_model.encode([question])
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

    # 💡 generowanie pomysłów (opcjonalne)
    idea_prompt_many = f"Temat: {answer_many}\nPomysły:\n"
    idea_prompt_one = f"Temat: {answer_one}\nPomysły:\n"

    ideas_output_many = idea_generator(
        idea_prompt_many,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        num_return_sequences=3
    )

    ideas_output_one = idea_generator(
        idea_prompt_one,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        num_return_sequences=3
    )

    ideas_many = []
    for o in ideas_output_many:
        text = o["generated_text"].replace(idea_prompt_many, "").strip()
        ideas_many.append(text.split("\n")[0])

    ideas_one = []
    for o in ideas_output_one:
        text = o["generated_text"].replace(idea_prompt_one, "").strip()
        ideas_one.append(text.split("\n")[0])

    # 📢 output
    print("\nAI_out:", answer_many)
    print("\nAI_in:", answer_one)

    if ideas_many:
        print("\n💡 Pomysły_zewnętrzne:")
        for i, idea in enumerate(ideas_many, 1):
            print(f"{i}. {idea}")

    if ideas_one:
        print("\n💡 Pomysły_wewnętrzne:")
        for i, idea in enumerate(ideas_one, 1):
            print(f"{i}. {idea}")

    print("\n" + "-"*50)