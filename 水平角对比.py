
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


def train_and_compare_with_azimuth_error_bins_final(orig_file: str, aug_file: str):
    """
    水平角分类版本：读取 azimuth（第 4 列），每 10° 一个类（共 36 类），
    统计预测误差的类别偏差（0类、1类、...）并绘图与输出，文案严谨。
    """
    receivers = np.array([
        [0, 0, 0],
        [36, 633, 2],
        [546, 596, 4],
        [297, 326, 6],
        [633, 70, 8]
    ])

    def compute_relative_angles(tx_pos, rx_positions):
        vectors = rx_positions - tx_pos
        horizontal_vectors = vectors[:, :2]
        norms = np.linalg.norm(horizontal_vectors, axis=1)
        norms[norms == 0] = 1e-6
        directions = horizontal_vectors / norms[:, None]
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        return np.degrees(angles) % 360

    def load_and_prepare_data(file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)

        tx_positions = df.iloc[:, 0:3].values
        rssi = df.iloc[:, 5:10].values
        azimuth_classes = (df.iloc[:, 3] // 10).astype(int).values
        angle_features = np.array([
            compute_relative_angles(tx, receivers) for tx in tx_positions
        ])
        X = np.hstack([rssi, angle_features])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return train_test_split(X_scaled, azimuth_classes, test_size=0.2, random_state=42)

    def train_and_get_class_diff(y_test, y_pred):
        return np.abs(y_pred - y_test)

    # 原始数据训练
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = load_and_prepare_data(orig_file)
    clf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = clf_orig.predict(X_test_orig)
    diff_orig = train_and_get_class_diff(y_test_orig, y_pred_orig)

    # 增强数据训练
    X_train_aug, X_test_aug, y_train_aug, y_test_aug = load_and_prepare_data(aug_file)
    clf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_aug.fit(X_train_aug, y_train_aug)
    y_pred_aug = clf_aug.predict(X_test_aug)
    diff_aug = train_and_get_class_diff(y_test_aug, y_pred_aug)

    # 分类偏差直方统计
    max_offset = 9
    bins = np.arange(0, max_offset + 1)
    x = np.arange(len(bins))
    counts_orig = np.array([(diff_orig == i).sum() for i in bins]) / len(diff_orig)
    counts_aug = np.array([(diff_aug == i).sum() for i in bins]) / len(diff_aug)

    bar_width = 0.35
    xtick_labels = [f"{i} class" for i in bins]

    # 画图
    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width / 2, counts_orig, width=bar_width, color="lightcoral", edgecolor="black", label="Original Data")
    plt.bar(x + bar_width / 2, counts_aug, width=bar_width, color="skyblue", edgecolor="black", label="Augmented Data")

    for i in range(len(bins)):
        plt.text(x[i] - bar_width/2, counts_orig[i] + 0.005, f"{counts_orig[i]*100:.1f}%", 
                 ha='center', va='bottom', fontsize=8, color='darkred')
        plt.text(x[i] + bar_width/2, counts_aug[i] + 0.005, f"{counts_aug[i]*100:.1f}%", 
                 ha='center', va='bottom', fontsize=8, color='navy')

    plt.xticks(x, xtick_labels)
    plt.ylim(0, 1.0)
    plt.yticks(np.linspace(0, 1.0, 11))
    plt.xlabel("Category Offset (|Predicted Class - True Class|)")
    plt.ylabel("Frequency")
    plt.title("Azimuth Angle Prediction Error by Category Offset")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 输出文案
    acc0_orig = np.mean(diff_orig == 0) * 100
    acc1_orig = np.mean(diff_orig <= 1) * 100
    acc2_orig = np.mean(diff_orig <= 2) * 100
    acc0_aug = np.mean(diff_aug == 0) * 100
    acc1_aug = np.mean(diff_aug <= 1) * 100
    acc2_aug = np.mean(diff_aug <= 2) * 100

    print("📊 Accuracy (Original Data):")
    print(f"✅ Exact match (0 class offset): {acc0_orig:.2f}%")
    print(f"✅ Within ±0 class (max ±10°): {acc1_orig:.2f}%")
    print(f"✅ Within ±2 classes (max ±30°): {acc2_orig:.2f}%\n")

    print("📊 Accuracy (Augmented Data):")
    print(f"✅ Exact match (0 class offset): {acc0_aug:.2f}%")
    print(f"✅ Within ±0 class (max ±10°): {acc1_aug:.2f}%")
    print(f"✅ Within ±2 classes (max ±30°): {acc2_aug:.2f}%")


train_and_compare_with_azimuth_error_bins_final(
    orig_file=r"d:\desktop\毕设材料\角度\angledata32303.csv",
    aug_file=r"D:\desktop\毕设材料\角度\angledata32303super.csv"
)
