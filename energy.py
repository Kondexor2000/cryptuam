import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def generate_energy_data(n=1000, random_state=42):
    np.random.seed(random_state)

    devices = np.random.randint(1, 10, n)
    hours = np.random.uniform(2, 16, n)
    efficiency = np.random.uniform(0.2, 1.0, n)
    eco_mode = np.random.randint(0, 2, n)

    energy_usage = (
        devices * hours * (1.5 - efficiency) * (1 - 0.3 * eco_mode)
        + np.random.normal(0, 2, n)
    )

    return pd.DataFrame({
        "devices": devices,
        "hours": hours,
        "efficiency": efficiency,
        "eco_mode": eco_mode,
        "energy_usage": energy_usage
    })


def split_energy_data(data):
    X = data.drop("energy_usage", axis=1)
    y = data["energy_usage"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_energy_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_energy_model(model, X_test, y_test):
    pred = model.predict(X_test)
    return mean_absolute_error(y_test, pred), pred


def simulate_energy_savings(model, X_test):
    X_test_saved = X_test.copy()
    X_test_saved["eco_mode"] = 1
    pred_before = model.predict(X_test)
    pred_after = model.predict(X_test_saved)
    return pred_before, pred_after, pred_before - pred_after


def emulate_one_device(model, devices=5, hours=10, efficiency=0.6):
    sample = pd.DataFrame([{
        "devices": devices,
        "hours": hours,
        "efficiency": efficiency,
        "eco_mode": 0
    }])
    classic = model.predict(sample)[0]
    co2 = model.predict(sample.assign(eco_mode=1))[0]
    return classic, co2


def main():
    data = generate_energy_data()
    X_train, X_test, y_train, y_test = split_energy_data(data)

    model = build_energy_model(X_train, y_train)
    mae, _ = evaluate_energy_model(model, X_test, y_test)
    print(f"Błąd średni absolutny (MAE): {mae:.2f} kWh")

    _, _, savings = simulate_energy_savings(model, X_test)
    print(f"Średnia: {np.mean(savings):.2f} kWh")
    print(f"Suma: {np.sum(savings):.2f} kWh")

    classic, co2 = emulate_one_device(model)
    print("\nSymulacja oszczędności:")
    print(f"Tradycyjny: {classic:.2f} kWh")
    print(f"Nowoczesny:   {co2:.2f} kWh")
    print(f"Różnica: {classic - co2:.2f} kWh")

    return {
        'model': model,
        'mae': mae,
        'savings': savings,
        'classic': classic,
        'co2': co2,
    }


if __name__ == "__main__":
    main()
