import pandas as pd
import glob
import os

# Folder that has all of your yearly CSV files in it
DATA_FOLDER = "C:\Python lessons"

#   Look for all of the XSV files
files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
print(f"Found {len(files)} files to combine.")

# Combine them
df_list = []
for file in files:
    temp = pd.read_csv(file)
    # keep reference
    temp["Source_File"] = os.path.basename(file)
    df_list.append(temp)

# Add all together
combined_df = pd.concat(df_list, ignore_index=True)
print("Combined shape:", combined_df.shape)

# sort by date if Month column exists
if "Month" in combined_df.columns:
    combined_df["Month"] = pd.to_datetime(
        combined_df["Month"], errors="coerce")
    combined_df = combined_df.sort_values("Month")

# Save combined dataset
combined_df.to_csv("cambridgeshire_correct_org.csv", index=False)
print(" Combined dataset saved as 'cambridgeshire_correct_org.csv'")
