# =========================
# 1. IMPORTY
# =========================
import wfdb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def load_ecg_data(record_name='100', pn_dir='mitdb'):
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    annotation = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)
    signal = record.p_signal[:, 0]
    peaks = annotation.sample
    labels = annotation.symbol
    return signal, peaks, labels


def extract_beats(signal, peaks, labels, window_size=200):
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

    return np.array(X), np.array(y)


def prepare_binary_labels(y):
    return np.array([0 if label == 'N' else 1 for label in y])


def train_ecg_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


model = None


def predict_beat(beat_signal, model=None):
    """
    beat_signal: numpy array o długości 200
    """
    if model is None:
        if globals().get('model') is None:
            raise ValueError('Model is not trained yet')
        model = globals()['model']

    beat_signal = beat_signal.reshape(1, -1)
    prediction = model.predict(beat_signal)[0]
    return "Normalne" if prediction == 0 else "Arytmia"


def main(show_plot=False):
    global model

    print("Pobieranie danych...")
    signal, peaks, labels = load_ecg_data()

    print(f"Długość sygnału: {len(signal)}")
    print(f"Liczba uderzeń: {len(peaks)}")

    print("Segmentacja sygnału...")
    X, y = extract_beats(signal, peaks, labels)
    print("Shape danych:", X.shape)

    print("Przygotowanie etykiet...")
    y_binary = prepare_binary_labels(y)
    print("Przykładowe etykiety:", y[:10])
    print("Binary:", y_binary[:10])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    print("Trenowanie modelu...")
    model = train_ecg_model(X_train, y_train)

    print("Ewaluacja...")
    y_pred = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("Wizualizacja przykładu...")
    idx = 0
    plt.figure(figsize=(8, 4))
    plt.plot(X[idx])
    plt.title(f"Label: {y[idx]} | Binary: {y_binary[idx]}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid()
    if show_plot:
        plt.show()

    import joblib
    file = "ecg_model.pkl"
    joblib.dump(model, file)
    print("Model zapisany jako ecg_model.pkl")

    print("\nTest predykcji:", predict_beat(X[0], model))
    return {
        'model': model,
        'X': X,
        'y': y,
        'y_binary': y_binary,
        'X_test': X_test,
        'y_test': y_test,
    }


if __name__ == "__main__":
    main()
