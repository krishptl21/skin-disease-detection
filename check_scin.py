import pandas as pd

cases = pd.read_csv("dataset/scin/scin_cases.csv")
labels = pd.read_csv("dataset/scin/scin_labels.csv")

scin = cases.merge(labels, on="case_id", how="inner")

print("Merged rows:", len(scin))
print()

print("IMPORTANT COLUMNS PREVIEW:")
print(scin[["case_id", "image_1_path", "image_2_path", "image_3_path", "weighted_skin_condition_label"]].head(10))
print()

print("UNIQUE SAMPLE LABEL VALUES:")
for i, val in enumerate(scin["weighted_skin_condition_label"].dropna().head(20)):
    print(f"{i+1}. {val}")