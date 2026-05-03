import os
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import numpy as np

# Załaduj model embeddingów z TensorFlow Hub
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

docs = []
for file in os.listdir("docs"):
    with open(f"docs/{file}", "r", encoding="utf-8") as f:
        docs.append(f.read())

# Generowanie embeddingów
embeddings = model(docs).numpy()

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, "docs.index")
np.save("docs.npy", docs)

print("Index created")