import os
import json

from pqcrypto.kem.ml_kem_512 import generate_keypair, encrypt

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# =========================================
# 1️⃣ Tworzenie dumpa bazy plików
# =========================================

docs = {}

for file in os.listdir("docs"):
    path = os.path.join("docs", file)

    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            docs[file] = f.read()

# zamiana na JSON bytes
data = json.dumps(docs).encode("utf-8")


# =========================================
# 2️⃣ ML-KEM (Post-Quantum Key Exchange)
# =========================================

public_key, secret_key = generate_keypair()

kem_ciphertext, shared_secret = encrypt(public_key)


# =========================================
# 3️⃣ HKDF – rozdzielenie kluczy
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
    "kem_ciphertext": kem_ciphertext.hex(),
    "hmac": mac.hex()
}

with open("backup_meta.json", "w") as f:
    json.dump(meta, f, indent=2)


print("Secure backup created successfully.")