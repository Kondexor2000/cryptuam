# =========================
# 1. IMPORTY
# =========================
import wfdb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# 2. POBRANIE DANYCH (MIT-BIH)
# =========================
print("Pobieranie danych...")

record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

signal = record.p_signal[:, 0]  # pierwszy kanał EKG
peaks = annotation.sample       # pozycje uderzeń
labels = annotation.symbol      # etykiety (N, V, itd.)

print(f"Długość sygnału: {len(signal)}")
print(f"Liczba uderzeń: {len(peaks)}")


# =========================
# 3. SEGMENTACJA (wycinanie uderzeń)
# =========================
print("Segmentacja sygnału...")

window_size = 200  # liczba próbek na jedno uderzenie

X = []
y = []

for i in range(1, len(peaks) - 1):
    peak = peaks[i]

    start = peak - window_size // 2
    end = peak + window_size // 2

    if start >= 0 and end < len(signal):
        beat = signal[start:end]
        X.append(beat)
        y.append(labels[i])

X = np.array(X)
y = np.array(y)

print("Shape danych:", X.shape)


# =========================
# 4. PRZYGOTOWANIE ETYKIET (binary classification)
# =========================
print("Przygotowanie etykiet...")

# 0 = normalne, 1 = arytmia
y_binary = np.array([0 if label == 'N' else 1 for label in y])

print("Przykładowe etykiety:", y[:10])
print("Binary:", y_binary[:10])


# =========================
# 5. PODZIAŁ NA TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)


# =========================
# 6. TRENING MODELU
# =========================
print("Trenowanie modelu...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 7. EWALUACJA
# =========================
print("Ewaluacja...")

y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))


# =========================
# 8. WIZUALIZACJA PRZYKŁADOWEGO UDERZENIA
# =========================
print("Wizualizacja przykładu...")

idx = 0

plt.figure(figsize=(8, 4))
plt.plot(X[idx])
plt.title(f"Label: {y[idx]} | Binary: {y_binary[idx]}")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


# =========================
# 9. ZAPIS MODELU
# =========================
import joblib

joblib.dump(model, "ecg_model.pkl")
print("Model zapisany jako ecg_model.pkl")


# =========================
# 10. PROSTA FUNKCJA PREDYKCJI
# =========================
def predict_beat(beat_signal):
    """
    beat_signal: numpy array o długości 200
    """
    beat_signal = beat_signal.reshape(1, -1)
    prediction = model.predict(beat_signal)[0]
    return "Normalne" if prediction == 0 else "Arytmia"


# test funkcji
print("\nTest predykcji:", predict_beat(X[0]))