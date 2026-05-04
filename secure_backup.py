import os
import json
import base64

from pqcrypto.kem.ml_kem_512 import generate_keypair, encrypt

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# =========================================
# 1️⃣ Wczytanie danych (plik lub folder)
# =========================================

INPUT_PATH = "ecg_model.pkl"  # może być plik LUB katalog

docs = {}

if os.path.isfile(INPUT_PATH):
    # pojedynczy plik
    with open(INPUT_PATH, "rb") as f:
        docs[os.path.basename(INPUT_PATH)] = base64.b64encode(f.read()).decode("utf-8")

elif os.path.isdir(INPUT_PATH):
    # katalog
    for file in os.listdir(INPUT_PATH):
        path = os.path.join(INPUT_PATH, file)

        if os.path.isfile(path):
            with open(path, "rb") as f:
                docs[file] = base64.b64encode(f.read()).decode("utf-8")

else:
    raise ValueError(f"Ścieżka nie istnieje: {INPUT_PATH}")


# JSON → bytes
data = json.dumps(docs).encode("utf-8")


# =========================================
# 2️⃣ ML-KEM (Post-Quantum)
# =========================================

public_key, secret_key = generate_keypair()

kem_ciphertext, shared_secret = encrypt(public_key)


# =========================================
# 3️⃣ HKDF – podział kluczy
# =========================================

hkdf = HKDF(
    algorithm=hashes.SHA256(),
    length=64,
    salt=None,
    info=b"secure-backup-system"
)

key_material = hkdf.derive(shared_secret)

aes_key = key_material[:32]
hmac_key = key_material[32:]


# =========================================
# 4️⃣ AES-256-CBC szyfrowanie
# =========================================

iv = os.urandom(16)

padder = padding.PKCS7(128).padder()
padded_data = padder.update(data) + padder.finalize()

cipher = Cipher(
    algorithms.AES(aes_key),
    modes.CBC(iv)
)

encryptor = cipher.encryptor()
ciphertext = encryptor.update(padded_data) + encryptor.finalize()


# =========================================
# 5️⃣ HMAC (integralność)
# =========================================

h = hmac.HMAC(hmac_key, hashes.SHA256())
h.update(iv + ciphertext)
mac = h.finalize()


# =========================================
# 6️⃣ Zapis backupu
# =========================================

with open("backup.enc", "wb") as f:
    f.write(iv + ciphertext)

meta = {
    "kem_ciphertext": base64.b64encode(kem_ciphertext).decode(),
    "hmac": base64.b64encode(mac).decode()
}

with open("backup_meta.json", "w") as f:
    json.dump(meta, f, indent=2)


print("Secure backup created successfully.")