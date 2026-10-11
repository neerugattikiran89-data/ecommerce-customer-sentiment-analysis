
import pandas as pd
from pathlib import Path

# Locate the project folder
project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "dataset" / "raw"
processed_folder = project_folder / "dataset" / "processed"

# Find the original dataset
csv_files = list(raw_folder.rglob("Reviews.csv"))

if not csv_files:
    print("Reviews.csv not found inside dataset/raw.")
else:
    print("Loading dataset...")

    # Load only the columns needed for sentiment analysis
    df = pd.read_csv(csv_files[0], usecols=["Score", "Text"])

    print("Original rows:", len(df))

    # Remove rows without review text or rating
    df = df.dropna(subset=["Score", "Text"])

    # Remove reviews containing only whitespace
    df = df[df["Text"].str.strip().ne("")]

    # Convert ratings into binary sentiment labels
    df["sentiment"] = df["Score"].apply(
        lambda score: 0 if score <= 2
        else (1 if score >= 4 else None)
    )

    # Exclude neutral 3-star reviews
    df = df.dropna(subset=["sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(int)

    # Remove exact duplicate review-text and rating combinations
    df = df.drop_duplicates(subset=["Text", "Score"])

    # Keep only the columns required for our first model
    df = df[["Text", "sentiment"]]

    # Save the processed dataset
    processed_folder.mkdir(parents=True, exist_ok=True)
    output_path = processed_folder / "reviews_processed.csv"
    df.to_csv(output_path, index=False)

    print("\nPreprocessing completed!")
    print("Processed rows:", len(df))
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts().sort_index())
    print("\nSaved to:", output_path)
