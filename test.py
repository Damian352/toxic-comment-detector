import pickle
from pathlib import Path

model_path = Path("models/model.pkl")
with model_path.open("rb") as f:
    model = pickle.load(f)

text = "I don't think you're right"
proba = model.predict_proba([text])

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
scores = {label: float(proba[0][i]) for i, label in enumerate(labels)}
print(scores)