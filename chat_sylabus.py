import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from rczar import secure_backup
import os

# ----------------------------
# 1️⃣ TensorFlow embedding model
# ----------------------------
url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(url)

# ----------------------------
# 2️⃣ FAISS + docs
# ----------------------------
index_file = "docs_sylabus.index"
npy_file = "docs_sylabus.npy"

index = faiss.read_index(index_file)
docs = np.load(npy_file, allow_pickle=True)

# ----------------------------
# 3️⃣ QA model (TensorFlow)
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
# 4️⃣ Idea model (TensorFlow GPT2)
# ----------------------------
IDEA_MODEL = "radlab/polish-gpt2-small-v2"

idea_tokenizer = AutoTokenizer.from_pretrained(IDEA_MODEL)
idea_model = AutoModelForCausalLM.from_pretrained(IDEA_MODEL)

if idea_tokenizer.pad_token is None:
    idea_tokenizer.pad_token = idea_tokenizer.eos_token

idea_generator = pipeline(
    "text-generation",
    model=idea_model,
    tokenizer=idea_tokenizer
)

def answer_question(question):
    q_embedding = embed_model([question]).numpy().astype("float32")
    faiss.normalize_L2(q_embedding)

    D, I = index.search(q_embedding, k=5)
    context_many = "\n".join([docs[i] for i in I[0]])
    context_one = docs[I[0][0]]

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

    answer_many = qa_generator(prompt_many)[0]["generated_text"]
    answer_one = qa_generator(prompt_one)[0]["generated_text"]

    idea_prompt_many = f"Temat: {answer_many}\nPomysły:\n"
    idea_prompt_one = f"Temat: {answer_one}\nPomysły:\n"

    ideas_many = idea_generator(
        idea_prompt_many,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        num_return_sequences=3
    )

    ideas_one = idea_generator(
        idea_prompt_one,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        num_return_sequences=3
    )

    return {
        "answer_many": answer_many.strip(),
        "answer_one": answer_one.strip(),
        "secure_answer_many": secure_backup(answer_many.strip().encode("utf-8")),
        "secure_answer_one": secure_backup(answer_one.strip().encode("utf-8")),
        "ideas_many": [
            o["generated_text"].replace(idea_prompt_many, "").strip().split("\n")[0]
            for o in ideas_many
        ],
        "ideas_one": [
            o["generated_text"].replace(idea_prompt_one, "").strip().split("\n")[0]
            for o in ideas_one
        ],
    }


# ----------------------------
# 6️⃣ Chat loop
# ----------------------------

def main():
    print("Mini ChatGPT (TensorFlow + sylabus). Wpisz 'exit'\n")

    while True:
        question = input("Ty: ")

        if question.lower() in ["exit", "quit"]:
            break

        result = answer_question(question)

        # ----------------------------
        # output
        # ----------------------------
        print("\nAI_out:", result["secure_answer_many"])
        print("\nAI_in:", result["secure_answer_one"])

        print("\nAI_out:", result["answer_many"])
        print("\nAI_in:", result["answer_one"])

        print("\n💡 Pomysły_zewnętrzne:")
        for i, idea in enumerate(result["ideas_many"], 1):
            print(i, secure_backup(idea.encode("utf-8")))

        print("\n💡 Pomysły_wewnętrzne:")
        for i, idea in enumerate(result["ideas_one"], 1):
            print(i, secure_backup(idea.encode("utf-8")))

        print("\n" + "-"*50)


if __name__ == "__main__":
    main()
