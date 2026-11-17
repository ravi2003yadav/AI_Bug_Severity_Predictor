
import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path

# 1. Load the cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_bug_reports.csv"
)

print("Original dataset:", df.shape)

# 2. Separate training and testing datasets
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Severity Label"]
)

# 3. Create output folder
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# 4. Save training dataset
train_df.to_csv(
    output_folder / "train_data.csv",
    index=False
)

# 5. Save testing dataset
test_df.to_csv(
    output_folder / "test_data.csv",
    index=False
)

# 6. Display dataset sizes
print("\nTRAINING DATASET SIZE:")
print(train_df.shape)

print("\nTESTING DATASET SIZE:")
print(test_df.shape)

# 7. Display severity distribution
print("\nTRAINING SEVERITY DISTRIBUTION:")
print(
    train_df["Severity Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTESTING SEVERITY DISTRIBUTION:")
print(
    test_df["Severity Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nDataset splitting completed successfully!")
