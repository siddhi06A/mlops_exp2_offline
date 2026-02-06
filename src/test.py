import joblib
import pandas as pd

# Load the model saved by train.py
model = joblib.load("iris_model.pkl")

# Create one sample to test (measurements for a Setosa flower)
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# Predict the species
pred = model.predict(sample)[0]
print("Test prediction:", pred)