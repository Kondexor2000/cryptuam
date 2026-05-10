import unittest

import numpy as np

import energy


class EnergyModuleTests(unittest.TestCase):
    def test_generate_energy_data_returns_dataframe(self):
        data = energy.generate_energy_data(n=100)
        self.assertEqual(len(data), 100)
        self.assertIn("energy_usage", data.columns)

    def test_split_energy_data_preserves_shape(self):
        data = energy.generate_energy_data(n=100)
        X_train, X_test, y_train, y_test = energy.split_energy_data(data)
        self.assertEqual(len(X_train) + len(X_test), 100)
        self.assertEqual(len(y_train) + len(y_test), 100)

    def test_build_and_evaluate_model(self):
        data = energy.generate_energy_data(n=100)
        X_train, X_test, y_train, y_test = energy.split_energy_data(data)
        model = energy.build_energy_model(X_train, y_train)
        mae, pred = energy.evaluate_energy_model(model, X_test, y_test)
        self.assertGreaterEqual(mae, 0)
        self.assertEqual(pred.shape[0], len(X_test))

    def test_simulate_energy_savings_returns_arrays(self):
        data = energy.generate_energy_data(n=100)
        X_train, X_test, y_train, y_test = energy.split_energy_data(data)
        model = energy.build_energy_model(X_train, y_train)
        pred_before, pred_after, savings = energy.simulate_energy_savings(model, X_test)
        self.assertEqual(pred_before.shape, pred_after.shape)
        self.assertEqual(pred_before.shape, savings.shape)

    def test_emulate_one_device_outputs_two_values(self):
        data = energy.generate_energy_data(n=100)
        X_train, X_test, y_train, y_test = energy.split_energy_data(data)
        model = energy.build_energy_model(X_train, y_train)
        classic, co2 = energy.emulate_one_device(model)
        self.assertIsInstance(classic, float)
        self.assertIsInstance(co2, float)


if __name__ == "__main__":
    unittest.main()
