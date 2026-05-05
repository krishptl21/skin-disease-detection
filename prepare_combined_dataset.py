import os
import ast
import pandas as pd

HAM_META = "dataset/ham10000/metadata.csv"
HAM_IMG_DIR = "dataset/ham10000/images"

SCIN_CASES = "dataset/scin/scin_cases.csv"
SCIN_LABELS = "dataset/scin/scin_labels.csv"

OUTPUT_CSV = "dataset/combined_metadata.csv"


# -----------------------------
# 1) Load HAM10000
# -----------------------------
ham = pd.read_csv(HAM_META)

ham_rows = []
for _, row in ham.iterrows():
    img_path = os.path.join(HAM_IMG_DIR, f"{row['image_id']}.jpg")
    if os.path.exists(img_path):
        ham_rows.append({
            "image_path": img_path,
            "label": row["dx"].strip().lower(),
            "source": "HAM10000"
        })

ham_df = pd.DataFrame(ham_rows)
print("HAM rows:", len(ham_df))


# -----------------------------
# 2) Load SCIN
# -----------------------------
cases = pd.read_csv(SCIN_CASES)
labels = pd.read_csv(SCIN_LABELS)

scin = cases.merge(labels, on="case_id", how="inner")
print("Merged SCIN rows:", len(scin))


# -----------------------------
# 3) Map SCIN label
# -----------------------------
def map_scin_label(weighted_label_str):
    if pd.isna(weighted_label_str):
        return None

    try:
        label_dict = ast.literal_eval(str(weighted_label_str))
    except:
        return None

    if not isinstance(label_dict, dict) or len(label_dict) == 0:
        return None

    top_label = max(label_dict.items(), key=lambda x: x[1])[0].lower().strip()

    if "eczema" in top_label or "dermatitis" in top_label:
        return "eczema"
    elif "psoriasis" in top_label:
        return "psoriasis"
    elif "acne" in top_label:
        return "acne"
    elif "tinea" in top_label or "fungal" in top_label or "candid" in top_label or "ringworm" in top_label:
        return "fungal"
    elif "basal cell carcinoma" in top_label:
        return "bcc"
    elif "melanoma" in top_label:
        return "mel"
    else:
        return None


# -----------------------------
# 4) Fix SCIN image path
# -----------------------------
def fix_scin_path(raw_path):
    if pd.isna(raw_path):
        return None

    raw_path = str(raw_path).replace("\\", "/").strip()

    if raw_path.startswith("dataset/images/"):
        filename = raw_path.replace("dataset/images/", "", 1)
        fixed_path = os.path.join("dataset", "scin", "images", filename)
        return fixed_path

    return raw_path


# -----------------------------
# 5) Extract SCIN rows
# -----------------------------
scin_rows = []

for _, row in scin.iterrows():
    final_label = map_scin_label(row.get("weighted_skin_condition_label"))

    if final_label is None:
        continue

    for col in ["image_1_path", "image_2_path", "image_3_path"]:
        if col in row.index and pd.notna(row[col]):
            img_path = fix_scin_path(row[col])

            if img_path and os.path.exists(img_path):
                scin_rows.append({
                    "image_path": img_path,
                    "label": final_label,
                    "source": "SCIN"
                })

scin_df = pd.DataFrame(scin_rows)
print("SCIN rows kept:", len(scin_df))


# -----------------------------
# 6) Combine and save
# -----------------------------
combined = pd.concat([ham_df, scin_df], ignore_index=True)

final_labels = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc",
    "eczema", "psoriasis", "acne", "fungal"
]
combined = combined[combined["label"].isin(final_labels)]

combined.to_csv(OUTPUT_CSV, index=False)

print("\nCombined dataset saved to:", OUTPUT_CSV)
print("\nSource counts:")
print(combined["source"].value_counts())

print("\nClass counts:")
print(combined["label"].value_counts())