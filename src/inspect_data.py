
import pandas as pd
from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Look for Reviews.csv anywhere inside dataset/raw
raw_folder = project_folder / "dataset" / "raw"
csv_files = list(raw_folder.rglob("Reviews.csv"))

if not csv_files:
    print("Reviews.csv was not found inside dataset/raw.")
else:
    file_path = csv_files[0]
    print("Dataset location:", file_path)

    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())
