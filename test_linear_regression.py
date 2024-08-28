# test_iris_ml.py
import numpy as np
import unittest
from linear_regression import reg, height, weight  # Assuming model is accessible

class TestLinearRegressionModel(unittest.TestCase):
    def test_model_accuracy(self):
        # Check if the model's score is above a certain threshold
        self.assertGreater(reg.score(np.array(height), weight), 0.9)  # Example threshold

if __name__ == '__main__':
    unittest.main()
