import os
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def resolve_docs_dir():
    # Wpisany tekst użytkownika jest wyszukiwany w materiale speech.
    if os.path.isdir("docs_speech") and os.listdir("docs_speech"):
        return "docs_speech"
    if os.path.isdir("docs") and os.listdir("docs"):
        return "docs"
    if os.path.isdir("docs_not") and os.listdir("docs_not"):
        return "docs_not"
    return "docs_speech"


docs_dir = resolve_docs_dir()
docs = []
for file in os.listdir(docs_dir):
    if not file.endswith(".txt"):
        continue
    with open(f"{docs_dir}/{file}", "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text:
            docs.append(text)

# Generowanie embeddingów. Gdy model TF Hub nie może zostać pobrany lub
# lokalny cache jest uszkodzony, użyj TF-IDF, który jest już zależnością projektu.
try:
    import tensorflow_hub as hub
    url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(url)
    embeddings = model(docs).numpy()
except Exception as exc:
    print(f"Nie udało się załadować modelu TensorFlow Hub ({exc}). Używam TF-IDF.")
    embeddings = TfidfVectorizer().fit_transform(docs).toarray().astype(np.float32)

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

index_file = "docs.index"
npy_file = "docs.npy"

faiss.write_index(index, index_file)
np.save(npy_file, docs)

print("Index created")
