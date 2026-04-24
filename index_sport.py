import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Zamiast wczytywania plików:
docs = [
    "Drużyna A wygrała 2:1 z Drużyną B. Ostatnie 5 meczów: W W L D W. Kondycja: dobra, brak kontuzji.",
    "Drużyna B przegrała 1:2 z Drużyną A. Ostatnie 5 meczów: L L W D L. Kondycja: średnia, 2 kontuzje kluczowych zawodników.",
    "Drużyna A zremisowała 0:0 z Drużyną C. Forma stabilna, wysoka wydolność fizyczna.",
    "Drużyna B wygrała 3:0 z Drużyną C. Poprawa formy, lepsza defensywa.",
    "Drużyna A wygrała 4:2 z Drużyną D. Atak w świetnej formie.",
    "Drużyna B przegrała 0:1 z Drużyną D. Problemy w ofensywie, niska skuteczność."
]

embeddings = model.encode(docs)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, "docs_sport.index")

np.save("docs_sport.npy", docs)

print("Index created")