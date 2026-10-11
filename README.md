# E-commerce Customer Sentiment Analysis

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify customer product reviews as **Positive** or **Negative**.

Users can enter a review into a web application and receive a sentiment prediction through a Flask API.

## Objectives

- Clean and preprocess customer review data.
- Convert text into numerical features using TF-IDF.
- Train and compare sentiment classification models.
- Evaluate model performance using classification metrics.
- Build an interactive web application for sentiment prediction.

## Dataset

**Amazon Fine Food Reviews**

The original dataset contains 568,454 reviews.

During preprocessing:
- Missing and blank reviews were removed.
- Three-star reviews were excluded.
- Ratings of 1–2 were labeled Negative.
- Ratings of 4–5 were labeled Positive.
- Duplicate review records were removed.

The processed dataset contains 363,903 reviews.

*Note: Sentiment labels are derived from star ratings, not manually annotated.*

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Flask
- HTML and CSS
- Joblib

## Model Performance

The selected Logistic Regression model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 92.14% |
| Negative precision | 69% |
| Negative recall | 92% |
| Negative F1-score | 79% |
| Positive precision | 98% |
| Positive recall | 92% |
| Positive F1-score | 95% |

Balanced class weights were used to improve the detection of negative reviews.

## Application Features

- Enter customer reviews through a webpage.
- Predict positive or negative sentiment.
- Display results interactively.
- Access predictions through a Flask API.
- Clear the review input and result.

## Project Structure

```text
Ecommerce_Customer_Sentiment_Analysis/
├── dataset/
│   ├── raw/
│   └── processed/
├── src/
│   ├── app.py
│   ├── analyze_data.py
│   ├── inspect_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── compare_models.py
│   └── predict_sentiment.py
├── trained_models/
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── visualizations/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/neerugattikiran89-data/ecommerce-customer-sentiment-analysis.git
cd ecommerce-customer-sentiment-analysis
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the application

```powershell
python src/app.py
```

### 5. Open the webpage

Visit http://127.0.0.1:5000 in your browser.

Enter a product review and click **Analyze Sentiment**.

## Limitations

- Predictions may not correctly interpret sarcasm or complex context.
- The dataset focuses on Amazon fine-food reviews.
- Sentiment labels are inferred from star ratings.
- Model predictions are estimates and may be incorrect.

## Future Improvements

- Add sentiment distribution charts and rating analysis.
- Display prediction confidence scores.
- Improve model performance through further experimentation.
- Add automated tests.
- Deploy the application online.
