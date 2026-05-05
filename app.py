from flask import Flask, render_template, request, url_for
import numpy as np
import os
import uuid
import pandas as pd
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        fl = alpha * tf.pow(1.0 - y_pred, gamma) * ce
        return tf.reduce_sum(fl, axis=-1)
    return loss_fn

import gdown

MODEL_PATH = "skin_model.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/file/d/1tJ_P0sAJFkV7Z0c8FP5UCFhRvyIGRba7/view?usp=drive_link"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=0.25)})

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with open("combined_labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

metadata_df = pd.read_csv("dataset/combined_metadata.csv")
class_counts = metadata_df["label"].value_counts().to_dict()
source_counts = metadata_df.groupby(["label", "source"]).size().unstack(fill_value=0).to_dict(orient="index")

CANCER_CLASSES = {"mel", "bcc", "akiec"}
BENIGN_LESION_CLASSES = {"nv", "bkl", "df", "vasc"}
OTHER_DISEASE_CLASSES = {"eczema", "psoriasis", "acne", "fungal"}

class_names = {
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Malignant Melanoma',
    'nv': 'Melanocytic Nevus (Mole)',
    'vasc': 'Vascular Lesion',
    'eczema': 'Eczema / Dermatitis',
    'psoriasis': 'Psoriasis',
    'acne': 'Acne',
    'fungal': 'Fungal Infection'
}

disease_info = {
    'akiec': {"name": "Actinic Keratosis / Intraepithelial Carcinoma", "category": "Skin Cancer", "description": "A precancerous or early cancerous skin lesion often related to long-term sun exposure.", "symptoms": "Rough, scaly, crusted patch, usually on sun-exposed skin.", "treatment": "Cryotherapy, topical therapy, laser therapy, or excision.", "aid": "Consult a dermatologist. Sun protection is important.", "source_note": "Knowledge mainly supported from HAM10000 lesion classes."},
    'bcc': {"name": "Basal Cell Carcinoma", "category": "Skin Cancer", "description": "A common type of skin cancer that usually grows slowly.", "symptoms": "Pearly bump, non-healing sore, shiny patch, or lesion with rolled border.", "treatment": "Surgical removal, cryotherapy, radiation, or topical medications.", "aid": "Consult a dermatologist for confirmation and treatment.", "source_note": "Knowledge supported from HAM10000 and SCIN."},
    'mel': {"name": "Malignant Melanoma", "category": "Skin Cancer", "description": "A dangerous skin cancer that can spread quickly if untreated.", "symptoms": "Asymmetry, irregular border, color variation, increasing size, changing mole.", "treatment": "Surgical excision, immunotherapy, targeted therapy, chemotherapy.", "aid": "Seek urgent dermatologist/oncology evaluation.", "source_note": "Knowledge supported from HAM10000 and SCIN."},
    'nv': {"name": "Melanocytic Nevus (Mole)", "category": "Benign Skin Lesion", "description": "A common mole that is usually harmless.", "symptoms": "Stable brown or black spot, usually symmetrical.", "treatment": "Usually no treatment unless suspicious changes occur.", "aid": "Monitor using the ABCDE rule and consult if changes occur.", "source_note": "Knowledge mainly supported from HAM10000."},
    'bkl': {"name": "Benign Keratosis", "category": "Benign Skin Lesion", "description": "A non-cancerous skin growth.", "symptoms": "Waxy, rough, or slightly raised lesion.", "treatment": "Usually not needed unless irritated or cosmetically unwanted.", "aid": "Consult if irritation, bleeding, or rapid change occurs.", "source_note": "Knowledge mainly supported from HAM10000."},
    'df': {"name": "Dermatofibroma", "category": "Benign Skin Lesion", "description": "A benign fibrous skin nodule.", "symptoms": "Firm small bump, often on legs, may dimple when pinched.", "treatment": "Usually no treatment required.", "aid": "Consult if painful or changing.", "source_note": "Knowledge mainly supported from HAM10000."},
    'vasc': {"name": "Vascular Lesion", "category": "Benign Skin Lesion", "description": "A lesion related to blood vessels, often benign.", "symptoms": "Red, purple, or vascular-looking spot.", "treatment": "Observation, laser treatment, or minor procedure if needed.", "aid": "Consult if growing, painful, or bleeding.", "source_note": "Knowledge mainly supported from HAM10000."},
    'eczema': {"name": "Eczema / Dermatitis", "category": "Other Skin Disease", "description": "An inflammatory skin condition causing itchy, red, and dry skin.", "symptoms": "Itching, redness, dryness, scaling, irritation.", "treatment": "Moisturizers, trigger avoidance, topical steroids, medical creams.", "aid": "Consult a dermatologist if persistent or severe.", "source_note": "Knowledge mainly supported from SCIN."},
    'psoriasis': {"name": "Psoriasis", "category": "Other Skin Disease", "description": "A chronic inflammatory skin disease causing thick scaly plaques.", "symptoms": "Red scaly plaques, flaking, itching, irritation.", "treatment": "Topical therapy, phototherapy, and systemic medications.", "aid": "Consult a dermatologist for long-term management.", "source_note": "Knowledge mainly supported from SCIN."},
    'acne': {"name": "Acne", "category": "Other Skin Disease", "description": "A common skin condition caused by clogged pores and inflammation.", "symptoms": "Pimples, blackheads, whiteheads, inflamed bumps.", "treatment": "Topical creams, oral medicines, and skincare changes.", "aid": "Consult if severe, painful, or leaving scars.", "source_note": "Knowledge mainly supported from SCIN."},
    'fungal': {"name": "Fungal Infection", "category": "Other Skin Disease", "description": "A skin infection caused by fungi.", "symptoms": "Itchy rash, ring-like lesions, scaling, redness.", "treatment": "Topical or oral antifungal medicines.", "aid": "Keep the area clean and dry and consult a doctor.", "source_note": "Knowledge mainly supported from SCIN."}
}

ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".webp")
LOW_CONF_THRESHOLD = 0.40
MEL_ALERT_THRESHOLD = 0.25

def calculate_clinical_risk(age, gender, body_part, itching, pain, duration, history, top_label, top_confidence):
    risk_score = 0
    reasons = []
    if age >= 50:
        risk_score += 2
        reasons.append("Age is above 50.")
    if body_part in ["face", "scalp", "neck", "arm", "hand"]:
        risk_score += 1
        reasons.append("Lesion is on a commonly sun-exposed area.")
    if duration in ["6-12 months", ">1 year"]:
        risk_score += 2
        reasons.append("Lesion has been present for a long duration.")
    if pain == "yes":
        risk_score += 1
        reasons.append("Pain is present.")
    if history in ["growing", "changing color", "bleeding", "changing size"]:
        risk_score += 3
        reasons.append("History suggests suspicious change in lesion.")
    if top_label in ["mel", "bcc", "akiec"]:
        risk_score += 3
        reasons.append("Image model predicts a cancer-related class.")
    if top_confidence >= 0.70:
        risk_score += 1
        reasons.append("Model confidence is high.")
    if itching == "yes" and top_label in ["eczema", "psoriasis", "fungal"]:
        reasons.append("Itching is clinically consistent with inflammatory skin disease.")

    if risk_score >= 7:
        level = "High Clinical Risk"
        recommendation = "Please seek dermatologist consultation as early as possible."
    elif risk_score >= 4:
        level = "Moderate Clinical Risk"
        recommendation = "Medical review is recommended, especially if the lesion is changing."
    else:
        level = "Low to Mild Clinical Risk"
        recommendation = "Continue monitoring, but seek care if symptoms worsen or change develops."
    return risk_score, level, recommendation, reasons

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        clinical_age = request.form.get("age", "").strip()
        clinical_gender = request.form.get("gender", "").strip()
        clinical_body_part = request.form.get("body_part", "").strip()
        clinical_itching = request.form.get("itching", "").strip()
        clinical_pain = request.form.get("pain", "").strip()
        clinical_duration = request.form.get("duration", "").strip()
        clinical_history = request.form.get("history", "").strip()

        if not file or not file.filename:
            return render_template("index.html", error="No file selected.")

        safe_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)
        uploaded_image = url_for("static", filename=f"uploads/{unique_name}")

        try:
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            img_array = preprocess_input(img_array)
            preds = model.predict(img_array, verbose=0)[0].astype(float)
        except Exception as e:
            return render_template("index.html", error=f"Prediction failed: {str(e)}")

        sorted_idx = np.argsort(preds)[::-1]
        top_index = int(sorted_idx[0])
        top_label = labels[top_index]
        top_confidence = float(preds[top_index])
        info = disease_info[top_label]

        # Clinical analysis
        try: age_val = int(clinical_age)
        except: age_val = 0
        risk_score, risk_level, rec, reasons = calculate_clinical_risk(age_val, clinical_gender.lower(), clinical_body_part.lower(), clinical_itching.lower(), clinical_pain.lower(), clinical_duration, clinical_history.lower(), top_label, top_confidence)

        # Build results
        results = {f"{class_names[labels[i]]} ({labels[i].upper()})": float(preds[i]) for i in sorted_idx}
        top3 = [{"name": class_names[labels[i]], "short": labels[i].upper(), "prob": float(preds[i]), "category": disease_info[labels[i]]["category"]} for i in sorted_idx[:3]]
        
        cancer_prob = float(sum(preds[labels.index(lbl)] for lbl in CANCER_CLASSES if lbl in labels))
        benign_prob = float(sum(preds[labels.index(lbl)] for lbl in BENIGN_LESION_CLASSES if lbl in labels))
        other_prob = float(sum(preds[labels.index(lbl)] for lbl in OTHER_DISEASE_CLASSES if lbl in labels))

        # Warnings
        warning_low = f"Low confidence prediction ({top_confidence*100:.1f}%). Use a clearer image." if top_confidence < LOW_CONF_THRESHOLD else None
        mel_prob = float(preds[labels.index("mel")]) if "mel" in labels else 0
        warning_mel = f"Melanoma probability is {mel_prob*100:.1f}%. Consult urgently." if mel_prob >= MEL_ALERT_THRESHOLD else None

        # RENDER RESULT PAGE
        return render_template(
            "result.html",
            uploaded_image=uploaded_image,
            top_name=info["name"],
            top_confidence=top_confidence,
            description=info["description"],
            symptoms=info["symptoms"],
            treatment=info["treatment"],
            aid=info["aid"],
            source_note=info["source_note"],
            broad_category=info["category"],
            top3=top3,
            results=results,
            warning_low_conf=warning_low,
            warning_mel=warning_mel,
            dataset_count=class_counts.get(top_label, 0),
            ham_count=source_counts.get(top_label, {}).get("HAM10000", 0),
            scin_count=source_counts.get(top_label, {}).get("SCIN", 0),
            cancer_prob=cancer_prob,
            benign_prob=benign_prob,
            other_prob=other_prob,
            clinical_risk_score=risk_score,
            clinical_risk_level=risk_level,
            clinical_recommendation=rec,
            clinical_reasons=reasons
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)