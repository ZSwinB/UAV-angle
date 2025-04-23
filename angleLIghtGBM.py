import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# === STEP 1: Load dataset ===
file_path = r"D:\desktop\毕设材料\angledata32303.csv"
df = pd.read_csv(file_path, header=None)
df.columns = ['x', 'y', 'z', 'u_deg', 'v_deg', 'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']

# === STEP 2: Replace noisy RSSI with NaN ===
RSSI_threshold = -120
def apply_nan_mask(row):
    rssi_vals = row[5:].values
    return pd.Series(np.where(rssi_vals > RSSI_threshold, rssi_vals, np.nan))

df[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']] = df.apply(apply_nan_mask, axis=1)
print(f"✅ Replaced RSSI ≤ {RSSI_threshold} with NaN")

# === STEP 3: u classification (36 classes, 10° per class) ===
df['u_class'] = df['u_deg'].apply(lambda x: int(x // 10) % 36)

# === STEP 4: Prepare train/test sets ===
X = df[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values
y = df['u_class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === STEP 5: Train LightGBM ===
clf = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
clf.fit(X_train, y_train)

# === STEP 6: Predict and evaluate ===
y_pred = clf.predict(X_test)
acc_top1 = accuracy_score(y_test, y_pred)

# Top-3 Accuracy
y_proba = clf.predict_proba(X_test)
top3 = np.argsort(y_proba, axis=1)[:, -3:]
top3_correct = [y_test[i] in top3[i] for i in range(len(y_test))]
acc_top3 = np.mean(top3_correct)

print("\n🎯 LightGBM (u, 36-class) results:")
print(f"  → Top-1 accuracy: {acc_top1:.2%}")
print(f"  → Top-3 accuracy: {acc_top3:.2%}")

# === STEP 7: Plot scatter ===
def plot_scatter(y_true, y_pred, title, num_classes):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=8)
    plt.plot([0, num_classes], [0, num_classes], 'r--')
    plt.xlim(0, num_classes)
    plt.ylim(0, num_classes)
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_scatter(y_test, y_pred, "Azimuth Classification (u, 36 Classes)", 36)
