import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("data/iris.csv")

# Split features (X) and target (y)
X = df.drop("species", axis=1)
y = df["species"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Calculate and print accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model to a file
joblib.dump(model, "iris_model.pkl")