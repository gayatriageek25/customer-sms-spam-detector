import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_loader import load_sms_dataset
from src.preprocessing import clean_text

os.makedirs("models", exist_ok=True)

print("Loading dataset...")
df = load_sms_dataset()

df["message"] = df["message"].apply(clean_text)
df["target"] = (df["label"] == "spam").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["target"],
    test_size=0.20,
    random_state=42,
    stratify=df["target"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nEvaluation")
print("-" * 40)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Not Spam", "Spam"]
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "models/spam_detector.joblib")
print("\nModel saved to models/spam_detector.joblib")
