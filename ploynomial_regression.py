# linear_regression.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Data: height in feet and corresponding weight in pounds
x = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
y = [16, 25, 36, 49, 64, 81, 100]

# Step 2 - Fitting Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Step 4 - Linear Regression prediction
print(f"Predicted weight for height 11.0 feet: {lin_reg.predict([[11]])[0]}")

# Polynomial Regression
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=1, include_bias=False),
    LinearRegression(),
)
polynomial_regression.fit(x, y)

# Predicting for height 20.0 feet
X_height = [[20.0]]
target_predicted = polynomial_regression.predict(X_height)
print(f"Predicted weight for height {X_height[0][0]} feet: {target_predicted[0]}")
