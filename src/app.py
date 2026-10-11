
from pathlib import Path

import joblib
from flask import Flask, request, jsonify, render_template

project_folder = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(project_folder / "templates"),
    static_folder=str(project_folder / "static")
)

model_path = project_folder / "trained_models" / "logistic_regression_model.joblib"
vectorizer_path = project_folder / "trained_models" / "improved_tfidf_vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    review = data.get("review")

    if not isinstance(review, str) or not review.strip():
        return jsonify({"error": "Please enter a review."}), 400

    review_tfidf = vectorizer.transform([review])
    prediction = int(model.predict(review_tfidf)[0])

    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "review": review,
        "sentiment": sentiment
    })


if __name__ == "__main__":
    app.run(debug=True)
