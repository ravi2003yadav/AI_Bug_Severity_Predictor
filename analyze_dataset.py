
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/raw/bugzilla_reports.csv")

# 1. Calculate severity percentages
print("\n1. SEVERITY PERCENTAGES")

severity_percentages = (
    df["Severity Label"]
    .value_counts(normalize=True) * 100
)

print(severity_percentages.round(2))

# 2. Check duplicate records
print("\n2. DUPLICATE RECORDS")
print(df.duplicated().sum())

# 3. Check duplicate bug IDs
print("\n3. DUPLICATE BUG IDs")
print(df["Bug ID"].duplicated().sum())

# 4. Check unique project names
print("\n4. PROJECT DISTRIBUTION")
print(df["Project"].value_counts())

# 5. Check resolution status
print("\n5. RESOLUTION STATUS")
print(df["Resolution Status"].value_counts())

# 6. Analyse short description length
print("\n6. DESCRIPTION LENGTH")

df["description_length"] = (
    df["Short Description"].str.len()
)

print(df["description_length"].describe())
