
import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/bugzilla_reports.csv")

# 1. Count how many times each Bug ID occurs
id_counts = df["Bug ID"].value_counts()

print("\n1. FREQUENCY OF BUG IDs")
print(id_counts.value_counts().sort_index())

# 2. Identify Bug IDs with multiple severity labels
severity_counts = (
    df.groupby("Bug ID")["Severity Label"].nunique()
)

conflicting_ids = severity_counts[
    severity_counts > 1
].index

# 3. Extract all records for conflicting IDs
conflict_records = df[
    df["Bug ID"].isin(conflicting_ids)
].sort_values("Bug ID")

print("\n2. TOTAL CONFLICTING RECORDS")
print(len(conflict_records))

# 4. Display the first 30 conflicting records
print("\n3. CONFLICTING BUG REPORTS")

print(
    conflict_records[
        ["Project", "Bug ID", "Short Description",
         "Severity Label"]
    ].head(40).to_string(index=False)
)

# 5. Count the severity combinations
print("\n4. SEVERITY LABEL DISTRIBUTION IN CONFLICTS")

print(
    conflict_records.groupby("Bug ID")["Severity Label"]
    .apply(lambda x: ", ".join(sorted(x.unique())))
    .value_counts()
)

# 6. Save all conflicting records for detailed inspection
conflict_records.to_csv(
    "results/conflicting_bug_ids.csv",
    index=False
)
# Check whether repeated IDs belong to different projects

print("\n5. PROJECT COMBINATIONS FOR REPEATED IDs")

repeated_project_check = (
    df[df["Bug ID"].duplicated(keep=False)]
    .groupby("Bug ID")
    .agg(
        project_count=("Project", "nunique"),
        projects=("Project", lambda x: ", ".join(sorted(x.unique()))),
        report_count=("Bug ID", "size")
    )
)

print(
    repeated_project_check["project_count"]
    .value_counts()
    .sort_index()
)

print("\n6. REPEATED IDs WITH THEIR PROJECTS")

print(
    repeated_project_check.head(30).to_string()
)
print("\nConflicting records saved successfully.")
