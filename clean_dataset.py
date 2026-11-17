
import pandas as pd
from pathlib import Path

# ---------------------------------------
# 1. LOAD ORIGINAL DATASET
# ---------------------------------------

input_path = "data/raw/bugzilla_reports.csv"

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# ---------------------------------------
# 2. SELECT RELEVANT COLUMNS
# ---------------------------------------

df = df[
    [
        "Project",
        "Bug ID",
        "Short Description",
        "Severity Label"
    ]
].copy()

# ---------------------------------------
# 3. CLEAN TEXT
# ---------------------------------------

# Remove spaces at the beginning and end
df["Short Description"] = (
    df["Short Description"].str.strip()
)

# Convert text to lowercase
df["cleaned_description"] = (
    df["Short Description"].str.lower()
)

# Replace multiple spaces with a single space
df["cleaned_description"] = (
    df["cleaned_description"]
    .str.replace(r"\s+", " ", regex=True)
)

# ---------------------------------------
# 4. VALIDATE THE DATA
# ---------------------------------------

# Check for empty descriptions
empty_count = (
    df["cleaned_description"].eq("").sum()
)

print("\nEmpty descriptions:", empty_count)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check severity categories
print("\nSeverity distribution:")
print(df["Severity Label"].value_counts())

# ---------------------------------------
# 5. SAVE CLEANED DATASET
# ---------------------------------------

output_path = Path("data/processed/cleaned_bug_reports.csv")

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved at:", output_path)
print("Final dataset shape:", df.shape)
