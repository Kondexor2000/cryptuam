import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. GENEROWANIE DANYCH
# -----------------------------
np.random.seed(42)

n = 1000

# cechy:
# - liczba urządzeń
# - średni czas pracy (h/dzień)
# - efektywność energetyczna (0-1, gdzie 1 = bardzo oszczędne)
# - tryb eco (0/1)
devices = np.random.randint(1, 10, n)
hours = np.random.uniform(2, 16, n)
efficiency = np.random.uniform(0.2, 1.0, n)
eco_mode = np.random.randint(0, 2, n)

# rzeczywiste zużycie energii (kWh)
energy_usage = (
    devices * hours * (1.5 - efficiency) * (1 - 0.3 * eco_mode)
    + np.random.normal(0, 2, n)
)

data = pd.DataFrame({
    "devices": devices,
    "hours": hours,
    "efficiency": efficiency,
    "eco_mode": eco_mode,
    "energy_usage": energy_usage
})

# -----------------------------
# 2. PRZYGOTOWANIE DANYCH
# -----------------------------
X = data.drop("energy_usage", axis=1)
y = data["energy_usage"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. MODEL ML
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predykcje
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
print(f"Błąd średni absolutny (MAE): {mae:.2f} kWh")

# -----------------------------
# 4. SYMULACJA OSZCZĘDNOŚCI
# -----------------------------
# przykład: włączamy tryb eco dla wszystkich
X_test_saved = X_test.copy()
X_test_saved["eco_mode"] = 1

pred_before = model.predict(X_test)
pred_after = model.predict(X_test_saved)

savings = pred_before - pred_after

print(f"Średnia: {np.mean(savings):.2f} kWh")
print(f"Suma: {np.sum(savings):.2f} kWh")

# -----------------------------
# 5. EMULATOR JEDNEGO URZĄDZENIA
# -----------------------------
sample = pd.DataFrame([{
    "devices": 5,
    "hours": 10,
    "efficiency": 0.6,
    "eco_mode": 0
}])

classic = model.predict(sample)[0]
co2 = model.predict(sample.assign(eco_mode=1))[0]

print("\nSymulacja oszczędności:")
print(f"Tradycyjny: {classic:.2f} kWh")
print(f"Nowoczesny:   {co2:.2f} kWh")
print(f"Różnica: {classic - co2:.2f} kWh")