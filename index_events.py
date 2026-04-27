import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Zamiast wczytywania plików:
docs = [
    "2026-05-12: Konferencja AI w Warszawie. Tematy: machine learning, NLP, przyszłość technologii.",
    "2026-06-01: Festiwal Muzyki Elektronicznej w Berlinie. Wystąpią topowi DJ-e z Europy.",
    "2026-04-30: Maraton Krakowski. Tysiące uczestników, wydarzenie sportowe o zasięgu międzynarodowym.",
    "2026-07-20: Open'er Festival w Gdyni. Koncerty światowych gwiazd muzyki.",
    "2026-09-15: Targi Gier Komputerowych w Poznaniu. Premiery nowych tytułów i sprzętu gamingowego.",
    "2026-08-05: Letni Festiwal Filmowy w Wenecji. Pokazy premierowe i konkurs główny.",
    "2026-10-10: Kongres Startupów w Londynie. Networking, inwestorzy i innowacje.",
    "2026-11-22: Koncert symfoniczny w Filharmonii Narodowej w Warszawie. Muzyka klasyczna na żywo."
]

embeddings = model.encode(docs)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, "docs_events.index")

np.save("docs_events.npy", docs)

print("Index created")