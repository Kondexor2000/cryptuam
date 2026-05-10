import importlib.util
import sys
import types
import unittest
from unittest.mock import patch, mock_open

import numpy as np

MODULE_PATH = r"c:\Users\kondz\OneDrive\Pulpit\Moje Prace\ChatGPT\cardio.py"

class DummyRecord:
    p_signal = np.arange(500).reshape(-1, 1)

class DummyAnn:
    sample = np.array([50, 150, 250, 350, 450])
    symbol = np.array(["N", "V", "N", "N", "V"])


def load_cardio_module():
    sys.modules["wfdb"] = types.SimpleNamespace(
        rdrecord=lambda name, pn_dir=None: DummyRecord(),
        rdann=lambda name, ext, pn_dir=None: DummyAnn()
    )
    spec = importlib.util.spec_from_file_location("cardio", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CardioModuleTests(unittest.TestCase):
    def test_predict_beat_returns_label(self):
        module = load_cardio_module()

        class DummyModel:
            def predict(self, X):
                return np.array([0])

        result = module.predict_beat(np.zeros(200), model=DummyModel())
        self.assertEqual(result, "Normalne")

    def test_main_executes_and_returns_metrics(self):
        module = load_cardio_module()

        with patch("joblib.dump", return_value=None):
            with patch("builtins.print"):
                result = module.main(show_plot=False)

        self.assertIn("model", result)
        self.assertIn("X_test", result)
        self.assertGreater(result["X"].shape[0], 0)
        self.assertEqual(result["X"].shape[1], 200)
        self.assertTrue(hasattr(result["model"], "predict"))

    def test_extract_beats_produces_expected_shape(self):
        module = load_cardio_module()
        signal = np.arange(500).reshape(-1, 1)
        peaks = np.array([50, 150, 250, 350, 450])
        labels = np.array(["N", "V", "N", "N", "V"])

        X, y = module.extract_beats(signal, peaks, labels)
        self.assertEqual(X.shape[1], 200)
        self.assertEqual(len(y), X.shape[0])


if __name__ == "__main__":
    unittest.main()
