
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Locate project folders
project_folder = Path(__file__).resolve().parent.parent
data_path = project_folder / "dataset" / "processed" / "reviews_processed.csv"
model_folder = project_folder / "trained_models"
model_folder.mkdir(parents=True, exist_ok=True)

# Load processed reviews
print("Loading data...")
df = pd.read_csv(data_path)

X = df["Text"].astype(str)
y = df["sentiment"]

# Use the same split for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# Preserve words such as 'not' and 'never'
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train two models
models = {
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    results[name] = {
        "accuracy": accuracy_score(y_test, predictions),
        "model": model
    }

    print(f"\n{name} accuracy: {results[name]['accuracy']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Compare overall accuracy
print("\n===== MODEL COMPARISON =====")
for name, result in results.items():
    print(f"{name}: {result['accuracy']:.4f} accuracy")

# Save the Logistic Regression model and its matching vectorizer
joblib.dump(
    results["Logistic Regression"]["model"],
    model_folder / "logistic_regression_model.joblib"
)
joblib.dump(
    vectorizer,
    model_folder / "improved_tfidf_vectorizer.joblib"
)

print("\nLogistic Regression model and vectorizer saved.")
