
# Import the Pandas library
import pandas as pd

# Specify the location of our dataset
file_path = "data/raw/bugzilla_reports.csv"

# Read the CSV file 
df = pd.read_csv(file_path, encoding="utf-8-sig")

# Display dataset dimensions
print("\n1. DATASET SIZE")
print(df.shape)

# Display column names
print("\n2. COLUMN NAMES")
print(df.columns.tolist())

# Display first five records
print("\n3. FIRST FIVE RECORDS")
print(df.head())

# Display dataset information
print("\n4. DATASET INFORMATION")
df.info()

# Check missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum())

# Count each severity category
print("\n6. SEVERITY DISTRIBUTION")
print(df["Severity Label"].value_counts())
