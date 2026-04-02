import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
for file in os.listdir("docs"):
    with open(f"docs/{file}", "r", encoding="utf-8") as f:
        docs.append(f.read())

embeddings = model.encode(docs)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, "docs.index")

np.save("docs.npy", docs)

print("Index created")