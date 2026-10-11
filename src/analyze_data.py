
import pandas as pd
from pathlib import Path

# Locate the project and dataset
project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "dataset" / "raw"

csv_files = list(raw_folder.rglob("Reviews.csv"))

if not csv_files:
    print("Dataset not found!")
else:
    # Load only the columns needed for this analysis
    df = pd.read_csv(csv_files[0], usecols=["Score", "Text"])

    print("Total reviews:", len(df))

    print("\nRating distribution:")
    print(df["Score"].value_counts().sort_index())

    print("\nRating percentages:")
    print(
        (df["Score"].value_counts(normalize=True).sort_index() * 100)
        .round(2)
    )

    print("\nDuplicate review rows:", df.duplicated().sum())

    print("\nExample review for each rating:")
    for score in sorted(df["Score"].unique()):
        review = df.loc[df["Score"] == score, "Text"].iloc[0]
        print(f"\nRating {score}: {review[:200]}")
