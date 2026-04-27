import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1️⃣ Model embeddingów
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 2️⃣ Dane (wydarzenia)
# ----------------------------
index = faiss.read_index("docs_events.index")
docs = np.load("docs_events.npy", allow_pickle=True)

# ważne: używamy cosine similarity
faiss.normalize_L2(index.reconstruct_n(0, index.ntotal))

# ----------------------------
# 3️⃣ Funkcja rekomendacji
# ----------------------------
def recommend(user_input, top_k=5):
    # embedding użytkownika
    user_emb = model.encode([user_input])
    faiss.normalize_L2(user_emb)

    # wyszukiwanie
    D, I = index.search(user_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((docs[idx], score))

    return results

# ----------------------------
# 4️⃣ Interfejs
# ----------------------------
print("🎯 System rekomendacji wydarzeń")
print("Opisz co lubisz (np. 'koncerty muzyki elektronicznej latem w Polsce')\n")

while True:
    user_input = input("Ty: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    recommendations = recommend(user_input, top_k=5)

    print("\n📅 Rekomendowane wydarzenia:\n")

    for i, (event, score) in enumerate(recommendations, 1):
        print(f"{i}. {event}")
        print(f"   dopasowanie: {score:.4f}\n")

    print("-" * 50)