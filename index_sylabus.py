import os
import faiss
import numpy as np
import PyPDF2
import tensorflow_hub as hub

# ----------------------------
# 1️⃣ TensorFlow embedding model
# ----------------------------
embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def read_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

docs = []
for file in os.listdir("sylabus"):
    path = f"sylabus/{file}"

    if file.endswith(".pdf"):
        text = read_pdf(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    chunks = chunk_text(text)
    docs.extend(chunks)

# ----------------------------
# 2️⃣ embeddings (TF)
# ----------------------------
embeddings = embed_model(docs).numpy().astype("float32")

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "docs_sylabus.index")
np.save("docs_sylabus.npy", docs)

print("Index created (TensorFlow)")