
import joblib
from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the saved model and vectorizer
model_path = project_folder / "trained_models" / "logistic_regression_model.joblib"
vectorizer_path = project_folder / "trained_models" / "improved_tfidf_vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


def predict_sentiment(review):
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


print("===================================")
print(" E-commerce Customer Sentiment Analysis")
print("===================================")
print("Type 'exit' to stop.\n")

while True:
    review = input("Enter a customer review: ").strip()

    if review.lower() == "exit":
        print("Program stopped.")
        break

    if not review:
        print("Please enter a review.\n")
        continue

    sentiment = predict_sentiment(review)

    print("Predicted sentiment:", sentiment)
    print()
