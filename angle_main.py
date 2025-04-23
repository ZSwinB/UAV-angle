import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

# === STEP 1: Load dataset ===
file_path = r"D:\desktop\毕设材料\angledata32303.csv"
df = pd.read_csv(file_path, header=None)
df.columns = ['x', 'y', 'z', 'u_deg', 'v_deg', 'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']

# === STEP 2: Replace noisy RSSI with -150 ===
RSSI_threshold = -150
RSSI_placeholder = -150

def replace_noise(row):
    rssi_vals = row[5:].values
    return pd.Series(np.where(rssi_vals > RSSI_threshold, rssi_vals, RSSI_placeholder))

df[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']] = df.apply(replace_noise, axis=1)

# === STEP 3: Label u into 36 classes (10° per class) ===
df['u_class'] = df['u_deg'].apply(lambda x: int(x // 10) % 36)

# === STEP 4: Train/test split ===
X = df[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values
y = df['u_class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === STEP 5: Train RF classifier ===
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# === STEP 6: Predict and evaluate ===
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Top-1 accuracy
acc_top1 = accuracy_score(y_test, y_pred)

# Top-3 accuracy
top3 = np.argsort(y_proba, axis=1)[:, -3:]
top3_correct = [y_test[i] in top3[i] for i in range(len(y_test))]
acc_top3 = np.mean(top3_correct)

print("\n🎯 Random Forest (u, 36-class) results:")
print(f"  → Top-1 accuracy: {acc_top1:.2%}")
print(f"  → Top-3 accuracy: {acc_top3:.2%}")

# === STEP 7: Confusion Matrix ===
def plot_confusion_matrix(y_true, y_pred, title, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred, "Azimuth Classification (u, 36 Classes) - RF", 36)