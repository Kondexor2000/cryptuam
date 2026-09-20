import os
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import numpy as np


def resolve_docs_dir():
    if os.path.isdir("docs") and os.listdir("docs"):
        return "docs"
    if os.path.isdir("docs_not") and os.listdir("docs_not"):
        return "docs_not"
    return "docs"


# Załaduj model embeddingów z TensorFlow Hub

url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(url)

docs_dir = resolve_docs_dir()
docs = []
for file in os.listdir(docs_dir):
    with open(f"{docs_dir}/{file}", "r", encoding="utf-8") as f:
        docs.append(f.read())

# Generowanie embeddingów
embeddings = model(docs).numpy()
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

index_file = "docs.index"
npy_file = "docs.npy"

faiss.write_index(index, index_file)
np.save(npy_file, docs)

print("Index created")