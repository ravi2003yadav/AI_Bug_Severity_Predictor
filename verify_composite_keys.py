
import pandas as pd

# Load original dataset
df = pd.read_csv("data/raw/bugzilla_reports.csv")

# 1. Check repeated combinations of Project and Bug ID
print("\n1. DUPLICATE COMPOSITE KEYS")

duplicate_keys = df.duplicated(
    subset=["Project", "Bug ID"],
    keep=False
)

print("Duplicate composite-key rows:", duplicate_keys.sum())

# 2. Count severity labels for each unique report
severity_counts = (
    df.groupby(["Project", "Bug ID"])["Severity Label"]
    .nunique()
)

# 3. Find reports with conflicting severity labels
conflicts = severity_counts[severity_counts > 1]

print("\n2. CONFLICTING SEVERITY LABELS")
print("Number of conflicting reports:", len(conflicts))

# 4. Verify unique report identities
print("\n3. DATASET SUMMARY")
print("Total records:", len(df))
print("Unique composite keys:", df[["Project", "Bug ID"]].drop_duplicates().shape[0])

# 5. Display final verification
if duplicate_keys.sum() == 0 and len(conflicts) == 0:
    print("\nSUCCESS: Every report has a unique identity and one severity label.")
else:
    print("\nFurther investigation is required.")
