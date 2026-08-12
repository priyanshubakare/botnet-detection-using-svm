
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from joblib import dump
# import seaborn as sns
# import json
# sns.set()

# DATA_PATH = "new1.csv"
# MODEL_OUT = "svm_MODEL.joblib"
# SCALER_OUT = "svm_scaler.joblib"

# # 1) Load
# df = pd.read_csv(DATA_PATH)
# print("Loaded:", DATA_PATH, "shape:", df.shape)

# # Choose label column
# label_col = None
# if 'Result' in df.columns:
#     label_col = 'Result'
# else:
#     for col in df.columns:
#         if df[col].nunique(dropna=True) == 2:
#             label_col = col
#             break
# if label_col is None:
#     raise RuntimeError("No binary label found. Set label_col manually.")

# print("Using label column:", label_col)
# y = df[label_col].copy()

# # 2) Prepare features
# X = df.drop(columns=[label_col]).copy()
# X = X.fillna(0)
# for c in X.columns:
#     if X[c].dtype == object:
#         X[c] = pd.to_numeric(X[c].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce').fillna(0)
# X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
# print("Feature matrix shape:", X.shape)

# # 3) Train/val/test
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.17647, stratify=y_temp, random_state=42)
# print(f"Splits -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

# # 4) Scale
# scaler = StandardScaler()
# X_train_s = scaler.fit_transform(X_train)
# X_val_s = scaler.transform(X_val)
# X_test_s = scaler.transform(X_test)

# # 5) SVM via SGDClassifier with warm start / partial fit
# svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4,
#                     learning_rate='optimal', max_iter=1, tol=None,
#                     warm_start=True, random_state=42)

# epochs = 50
# train_accs = []
# val_accs = []
# hinge_losses = []

# def to_pm1(y):
#     y_arr = np.array(y)
#     uniq = np.unique(y_arr)
#     if set(uniq) <= {0,1}:
#         return np.where(y_arr==1, 1, -1)
#     return y_arr

# y_train_pm1 = to_pm1(y_train)

# for epoch in range(epochs):
#     if epoch == 0:
#         svm.partial_fit(X_train_s, y_train, classes=np.unique(y_train))
#     else:
#         svm.partial_fit(X_train_s, y_train)

#     train_pred = svm.predict(X_train_s)
#     val_pred = svm.predict(X_val_s)

#     train_acc = accuracy_score(y_train, train_pred)
#     val_acc = accuracy_score(y_val, val_pred)
#     train_accs.append(train_acc)
#     val_accs.append(val_acc)

#     margins = y_train_pm1 * svm.decision_function(X_train_s)
#     hinge = np.maximum(0, 1 - margins).mean()
#     hinge_losses.append(hinge)

#     print(f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, hinge_loss={hinge:.4f}")

# # 6) Plots
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(range(1, epochs+1), hinge_losses, marker='o')
# plt.title("Hinge Loss (training)")
# plt.xlabel("Epoch")
# plt.ylabel("Hinge loss")
# plt.grid(True)

# plt.subplot(1,2,2)
# plt.plot(range(1, epochs+1), train_accs, label='train_acc', marker='o')
# plt.plot(range(1, epochs+1), val_accs, label='val_acc', marker='o')
# plt.title("Accuracy (train vs val)")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # 7) Final evaluation on test set
# test_pred = svm.predict(X_test_s)
# print("\n=== Test set accuracy ===")
# print("Accuracy:", accuracy_score(y_test, test_pred))
# print("\n=== Classification report ===")
# print(classification_report(y_test, test_pred))

# print("\n=== Confusion matrix ===")
# cm = confusion_matrix(y_test, test_pred)
# plt.figure(figsize=(5,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("SVM (linear) Confusion Matrix on Test")
# plt.show()

# # 8) Save model + scaler + feature names
# feature_names = list(X.columns)
# with open("feature_names.json", "w", encoding="utf-8") as f:
#     json.dump(feature_names, f)
# dump(feature_names, "feature_names.joblib")
# dump(svm, MODEL_OUT)
# dump(scaler, SCALER_OUT)
# print(f"Saved model -> {MODEL_OUT}")
# print(f"Saved scaler -> {SCALER_OUT}")
# print("✅ Saved feature names -> feature_names.json and feature_names.joblib")



"""
svm_train_with_metrics.py

- Reads: new1.csv (expects a column named 'Result' for labels)
- Trains: linear SVM via SGDClassifier (hinge)
- Saves: svm_MODEL.joblib, svm_scaler.joblib, feature_names.json, feature_names.joblib, label_mapping.json
- Produces training/validation loss/accuracy plots and test confusion matrix + classification report
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set()

DATA_PATH = "new1.csv"           # put new1.csv here
MODEL_OUT = "svm_MODEL.joblib"
SCALER_OUT = "svm_scaler.joblib"
FEATURES_JSON = "feature_names.json"
FEATURES_JOBLIB = "feature_names.joblib"
LABEL_MAP = "label_mapping.json"

# -------------------------
# 1) Load data
# -------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH, "shape:", df.shape)
if "Result" not in df.columns:
    raise RuntimeError("Expected a 'Result' label column in the CSV")

# label
label_col = "Result"
y = df[label_col].copy()

# -------------------------
# 2) Features: keep exact order from CSV (all columns except label)
# -------------------------
X = df.drop(columns=[label_col]).copy()
# Fill NaNs with 0
X = X.fillna(0)

# Convert any non-numeric columns into numeric where possible (best-effort)
for c in X.columns:
    if X[c].dtype == object:
        # strip non-digit characters and try numeric; fallback to 0
        X[c] = pd.to_numeric(X[c].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce').fillna(0)

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
print("Feature matrix shape:", X.shape)
feature_names = list(X.columns)

# Save feature names (exact order)
with open(FEATURES_JSON, "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=2)
dump(feature_names, FEATURES_JOBLIB)
print(f"Saved feature names to {FEATURES_JSON}")

# -------------------------
# 3) Train / val / test split
# -------------------------
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.17647, stratify=y_temp, random_state=42)
print(f"Splits -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

# -------------------------
# 4) Scale features
# -------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# -------------------------
# 5) Setup SVM (linear via SGD)
# -------------------------
svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4,
                    learning_rate='optimal', max_iter=1, tol=None,
                    warm_start=True, random_state=42)

epochs = 50
train_accs = []
val_accs = []
hinge_losses = []

# convert labels to +/-1 for hinge loss
def to_pm1(y_series):
    y_arr = np.array(y_series)
    uniq = np.unique(y_arr)
    if set(uniq) <= {0,1}:
        return np.where(y_arr == 1, 1, -1)
    # if already -1/+1
    return y_arr

y_train_pm1 = to_pm1(y_train)

for epoch in range(epochs):
    if epoch == 0:
        svm.partial_fit(X_train_s, y_train, classes=np.unique(y_train))
    else:
        svm.partial_fit(X_train_s, y_train)

    train_pred = svm.predict(X_train_s)
    val_pred = svm.predict(X_val_s)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    margins = y_train_pm1 * svm.decision_function(X_train_s)
    hinge = np.maximum(0, 1 - margins).mean()
    hinge_losses.append(hinge)

    print(f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, hinge_loss={hinge:.4f}")

# -------------------------
# 6) Plots
# -------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), hinge_losses, marker='o')
plt.title("Hinge Loss (training)")
plt.xlabel("Epoch")
plt.ylabel("Hinge loss")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), train_accs, label='train_acc', marker='o')
plt.plot(range(1, epochs+1), val_accs, label='val_acc', marker='o')
plt.title("Accuracy (train vs val)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 7) Final evaluation on test set
# -------------------------
test_pred = svm.predict(X_test_s)
print("\n=== Test set accuracy ===")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("\n=== Classification report ===")
print(classification_report(y_test, test_pred))

print("\n=== Confusion matrix ===")
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM (linear) Confusion Matrix on Test")
plt.show()

# -------------------------
# 8) Save model + scaler
# -------------------------
dump(svm, MODEL_OUT)
dump(scaler, SCALER_OUT)
print(f"Saved model -> {MODEL_OUT}")
print(f"Saved scaler -> {SCALER_OUT}")

# -------------------------
# 9) Save label mapping (which numeric label corresponds to botnet)
#    Heuristic: choose the class with the higher average number of feature-1s per row
# -------------------------
try:
    X_num = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    row_sums = X_num.sum(axis=1)
    avg_by_label = {}
    for cls in np.unique(y):
        mask = (y == cls)
        avg_by_label[int(cls)] = float(row_sums[mask].mean()) if mask.sum() > 0 else 0.0

    # class with larger average sum likely represents botnet (more flags)
    botnet_value = int(max(avg_by_label.items(), key=lambda x: x[1])[0])

    mapping = {"botnet_value": botnet_value, "avg_feature_sum": avg_by_label}
    with open(LABEL_MAP, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print("Saved label mapping:", mapping)
except Exception as e:
    print("Failed to auto-create label mapping:", e)
    # fallback
    with open(LABEL_MAP, "w", encoding="utf-8") as f:
        json.dump({"botnet_value": 1}, f, indent=2)
    print("Saved fallback label_mapping.json with botnet_value=1")
