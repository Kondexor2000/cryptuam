import os
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import numpy as np

# Załaduj model embeddingów z TensorFlow Hub

url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(url)

docs = []
for file in os.listdir("docs"):
    with open(f"docs/{file}", "r", encoding="utf-8") as f:
        docs.append(f.read())

# Generowanie embeddingów
embeddings = model(docs).numpy()

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

index_file = "docs.index"
npy_file = "docs.npy"

faiss.write_index(index, index_file)
np.save(npy_file, docs)

print("Index created")