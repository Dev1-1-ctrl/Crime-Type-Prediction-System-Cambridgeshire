import pandas as pd

# Load the  dataset
df = pd.read_csv("cambridgeshire_correct_org.csv")

print("Initial shape:", df.shape)

# Drop useless columns
df = df.drop(columns=["Context", "Crime ID", "Month_raw"], errors="ignore")

#  Convert Month to datetime
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

# Extract time-based features
df["Year"] = df["Month"].dt.year
df["Month_num"] = df["Month"].dt.month
df["Season"] = df["Month"].dt.month % 12 // 3 + \
    1   # 1=Winter, 2=Spring, 3=Summer, 4=Autumn

# Clean text columns
for col in ["LSOA name", "Crime type"]:
    df[col] = df[col].astype(str).str.strip().str.title()

#  Handle missing Latitude/Longitude (drop if they are not in)
df = df.dropna(subset=["Latitude", "Longitude"])

#  Droping duplicates
df = df.drop_duplicates()

print("After cleaning:", df.shape)

# Save cleaned dataset
df.to_csv("2017 to 2025 cambridgeshire_final_cleaned.csv", index=False)
print(" Cleaned dataset saved as '2017 to 2025 cambridgeshire_final_cleaned.csv'")
print(df.head())
