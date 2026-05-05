import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


RESULTS_DIR = "results_combined"
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = "dataset/combined_metadata.csv"


def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )
        ce = -y_true * tf.math.log(y_pred)
        fl = alpha * tf.pow(1.0 - y_pred, gamma) * ce
        return tf.reduce_sum(fl, axis=-1)
    return loss_fn


# -----------------------------
# 1) Load metadata
# -----------------------------
metadata = pd.read_csv(CSV_PATH)

labels = sorted(metadata["label"].unique())
label_map = {label: idx for idx, label in enumerate(labels)}
metadata["target"] = metadata["label"].map(label_map)

print("Labels:", labels)
print("Label map:", label_map)
print("\nClass counts:")
print(metadata["label"].value_counts())


# -----------------------------
# 2) Load images
# -----------------------------
images = []
targets = []

for _, row in metadata.iterrows():
    img_path = row["image_path"]

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = preprocess_input(img)

    images.append(img)
    targets.append(row["target"])

X = np.array(images, dtype=np.float32)
y = np.array(targets, dtype=np.int32)

print("\nLoaded images:", X.shape)


# -----------------------------
# 3) Split data
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

y_train_cat = to_categorical(y_train, num_classes=len(labels))
y_val_cat = to_categorical(y_val, num_classes=len(labels))
y_test_cat = to_categorical(y_test, num_classes=len(labels))


# -----------------------------
# 4) Class weights
# -----------------------------
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_arr))

print("\nClass weights:")
print(class_weights)


# -----------------------------
# 5) Data generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=16, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val_cat, batch_size=16, shuffle=False)


# -----------------------------
# 6) Build model
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(len(labels), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)


# -----------------------------
# 7) Callbacks
# -----------------------------
callbacks1 = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6),
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, "best_phase1_combined.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

callbacks2 = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6),
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, "best_phase2_combined.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]


# -----------------------------
# 8) Phase 1
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nPhase 1 starting...\n")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=8,
    callbacks=callbacks1,
    class_weight=class_weights
)


# -----------------------------
# 9) Phase 2
# -----------------------------
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=["accuracy"]
)

print("\nPhase 2 starting...\n")
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12,
    callbacks=callbacks2,
    class_weight=class_weights
)


# -----------------------------
# 10) Evaluation
# -----------------------------
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred_classes, target_names=labels)
print("\nClassification Report:\n")
print(report)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)
disp.plot(ax=ax, xticks_rotation=45, values_format="d")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=200)
plt.close(fig)


# -----------------------------
# 11) Curves
# -----------------------------
def merge_hist(h1, h2, key):
    return list(h1.history.get(key, [])) + list(h2.history.get(key, []))

acc = merge_hist(history1, history2, "accuracy")
val_acc = merge_hist(history1, history2, "val_accuracy")
loss = merge_hist(history1, history2, "loss")
val_loss = merge_hist(history1, history2, "val_loss")

fig_acc = plt.figure()
plt.plot(acc)
plt.plot(val_acc)
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=200)
plt.close(fig_acc)

fig_loss = plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=200)
plt.close(fig_loss)


# -----------------------------
# 12) Save labels and model
# -----------------------------
with open("combined_labels.txt", "w", encoding="utf-8") as f:
    for label in labels:
        f.write(label + "\n")

model.save("skin_combined_model.keras")

print("\nSaved model -> skin_combined_model.keras")
print("Saved labels -> combined_labels.txt")
print("Saved outputs in -> results_combined/")