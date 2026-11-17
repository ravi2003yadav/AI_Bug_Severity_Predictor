
import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/bugzilla_reports.csv")

# --------------------------------
# 1. INVESTIGATE REPEATED BUG IDs
# --------------------------------

print("\n1. REPEATED BUG ID DETAILS")

# Find every row whose Bug ID appears more than once
repeated_ids = df[
    df["Bug ID"].duplicated(keep=False)
]

print("Total rows with repeated IDs:", len(repeated_ids))

# Display the first 20 repeated records
print(
    repeated_ids[
        ["Bug ID", "Short Description", "Severity Label"]
    ].head(20).to_string(index=False)
)

# Check whether repeated IDs have different severity labels
print("\n2. BUG IDs WITH DIFFERENT SEVERITY LABELS")

severity_counts = (
    df.groupby("Bug ID")["Severity Label"].nunique()
)

conflicting_ids = severity_counts[severity_counts > 1]

print("Number of IDs with conflicting labels:", len(conflicting_ids))

# --------------------------------
# 2. INVESTIGATE SHORT DESCRIPTIONS
# --------------------------------

print("\n3. VERY SHORT DESCRIPTIONS")

short_reports = df[
    df["Short Description"].str.strip().str.len() <= 5
]

print("Number of very short descriptions:", len(short_reports))

print(
    short_reports[
        ["Bug ID", "Short Description", "Severity Label"]
    ].head(20).to_string(index=False)
)

# --------------------------------
# 3. CHECK EMPTY OR WHITESPACE TEXT
# --------------------------------

print("\n4. EMPTY OR WHITESPACE DESCRIPTIONS")

empty_reports = df[
    df["Short Description"].str.strip().eq("")
]

print("Empty descriptions:", len(empty_reports))
