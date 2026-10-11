
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Locate project folders
project_folder = Path(__file__).resolve().parent.parent

data_path = (
    project_folder
    / "dataset"
    / "processed"
    / "reviews_processed.csv"
)

model_folder = project_folder / "trained_models"
model_folder.mkdir(parents=True, exist_ok=True)

# 2. Load processed data
print("Loading processed reviews...")
df = pd.read_csv(data_path)

X = df["Text"].astype(str)
y = df["sentiment"]

print("Total reviews:", len(df))
print("Preparing training and testing data...")

# 3. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# 4. Convert review text into numerical features
print("\nConverting text into TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train the model
print("Training Naive Bayes model...")

model = MultinomialNB(alpha=1.0)
model.fit(X_train_tfidf, y_train)

# 6. Predict sentiment for unseen test reviews
print("Evaluating model...")

y_pred = model.predict(X_test_tfidf)

# 7. Evaluate performance
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 8. Save model and vectorizer
joblib.dump(model, model_folder / "sentiment_model.joblib")
joblib.dump(vectorizer, model_folder / "tfidf_vectorizer.joblib")

print("\nModel and vectorizer saved successfully!")
print("Model folder:", model_folder)
