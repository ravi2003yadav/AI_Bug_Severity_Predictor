
# ==========================================
# CREATE TRAINING, VALIDATION AND TEST DATA
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split


# STEP 1: Load the cleaned dataset

df = pd.read_csv("data/processed/cleaned_bug_reports.csv")

print("Original dataset size:", df.shape)


# STEP 2: Separate training data (70%)

train_df, remaining_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Severity Label"]
)


# STEP 3: Divide remaining data into validation and testing (15% each)

validation_df, test_df = train_test_split(
    remaining_df,
    test_size=0.50,
    random_state=42,
    stratify=remaining_df["Severity Label"]
)


# STEP 4: Save all three datasets

train_df.to_csv(
    "data/processed/final_train.csv",
    index=False
)

validation_df.to_csv(
    "data/processed/validation.csv",
    index=False
)

test_df.to_csv(
    "data/processed/final_test.csv",
    index=False
)


# STEP 5: Display dataset sizes

print("\nFINAL DATASET SIZES")

print("Training:", train_df.shape)
print("Validation:", validation_df.shape)
print("Testing:", test_df.shape)


# STEP 6: Display severity distribution

print("\nTRAINING SEVERITY DISTRIBUTION")

print(
    train_df["Severity Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nVALIDATION SEVERITY DISTRIBUTION")

print(
    validation_df["Severity Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTESTING SEVERITY DISTRIBUTION")

print(
    test_df["Severity Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nThree-way dataset splitting completed!")
