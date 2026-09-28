import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load training data
data = pd.read_csv("data/training_data.csv")

# Separate input and output
X = data["text"]
y = data["risk"]

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# Train the model
model.fit(X, y)

# Save trained model
joblib.dump(
    model,
    "models/risk_classifier.pkl"
)

print("Model trained successfully!")
print("Saved as: models/risk_classifier.pkl")