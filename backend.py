from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

import cardio
import energy


HOST = "127.0.0.1"
PORT = 5000
BASE_DIR = Path(__file__).resolve().parent
API_TEST_PAGE = BASE_DIR / "api_test.html"


def json_response(payload, status=200):
    return status, {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8")


def html_response(content, status=200):
    return status, {"Content-Type": "text/html; charset=utf-8"}, content.encode("utf-8")


def error_response(message, status=400):
    return json_response({"error": message}, status)


@lru_cache(maxsize=1)
def get_energy_model():
    data = energy.generate_energy_data()
    X_train, X_test, y_train, y_test = energy.split_energy_data(data)
    model = energy.build_energy_model(X_train, y_train)
    mae, _ = energy.evaluate_energy_model(model, X_test, y_test)
    return model, float(mae)


@lru_cache(maxsize=1)
def get_cardio_model():
    try:
        import joblib

        return joblib.load("ecg_model.pkl")
    except Exception:
        result = cardio.main(show_plot=False)
        return result["model"]


def parse_float(value, field_name, minimum=None, maximum=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number")

    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")
    return parsed


def parse_int(value, field_name, minimum=None, maximum=None):
    parsed = parse_float(value, field_name, minimum, maximum)
    if not parsed.is_integer():
        raise ValueError(f"{field_name} must be an integer")
    return int(parsed)


def health():
    return json_response({"status": "ok", "projects": [1, 3, 4]})


def api_test_page():
    if not API_TEST_PAGE.exists():
        return error_response("api_test.html not found", 404)
    return html_response(API_TEST_PAGE.read_text(encoding="utf-8"))


def predict_ecg(data):
    beat = data.get("beat")
    if not isinstance(beat, list):
        return error_response("beat must be a list with 200 numeric samples")
    if len(beat) != 200:
        return error_response("beat must contain exactly 200 samples")

    try:
        beat_signal = np.array([float(sample) for sample in beat])
    except (TypeError, ValueError):
        return error_response("beat must contain only numbers")

    model = get_cardio_model()
    label = cardio.predict_beat(beat_signal, model=model)
    return json_response({"prediction": label})


def ask_syllabus(data):
    question = data.get("question", "").strip()
    if not question:
        return error_response("question is required")

    import chat_sylabus

    return json_response(chat_sylabus.answer_question(question))


def predict_energy(data):
    try:
        devices = parse_int(data.get("devices"), "devices", minimum=1)
        hours = parse_float(data.get("hours"), "hours", minimum=0)
        efficiency = parse_float(data.get("efficiency"), "efficiency", minimum=0, maximum=1)
        eco_mode = parse_int(data.get("eco_mode", 0), "eco_mode", minimum=0, maximum=1)
    except ValueError as exc:
        return error_response(str(exc))

    model, mae = get_energy_model()
    sample = pd.DataFrame([{
        "devices": devices,
        "hours": hours,
        "efficiency": efficiency,
        "eco_mode": eco_mode,
    }])
    predicted_usage = float(model.predict(sample)[0])
    classic, eco = energy.emulate_one_device(model, devices, hours, efficiency)

    return json_response({
        "predicted_kwh": round(predicted_usage, 2),
        "mae": round(mae, 2),
        "classic_kwh": round(float(classic), 2),
        "eco_kwh": round(float(eco), 2),
        "estimated_savings_kwh": round(float(classic - eco), 2),
    })


def route(method, path, data=None):
    if method == "GET" and path in ["/", "/api-test.html"]:
        return api_test_page()
    if method == "GET" and path == "/api/health":
        return health()
    if method == "POST" and path == "/api/project1/ecg/predict":
        return predict_ecg(data or {})
    if method == "POST" and path == "/api/project3/syllabus/ask":
        return ask_syllabus(data or {})
    if method == "POST" and path == "/api/project4/energy/predict":
        return predict_energy(data or {})
    return error_response("Endpoint not found", 404)


class PortfolioRequestHandler(BaseHTTPRequestHandler):
    def _send(self, response):
        status, headers, body = response
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        self._send(route("GET", path))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(error_response("Request body must be valid JSON"))
            return

        path = urlparse(self.path).path
        self._send(route("POST", path, data))


def run(host=HOST, port=PORT):
    server = HTTPServer((host, port), PortfolioRequestHandler)
    print(f"Backend running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
