import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_antenna_pattern(json_path, key="theta"):
    with open(json_path, 'r') as f:
        ant_data = json.load(f)
    pattern = ant_data[key]
    resolution = pattern["resolution"]
    real = np.array(pattern["real"])
    imag = np.array(pattern.get("imag", np.zeros_like(real)))
    gain_linear = np.sqrt(real**2 + imag**2)
    gain_db = 10 * np.log10(gain_linear + 1e-12)
    return gain_db, resolution


def apply_angular_offset(original_angle, offset_deg):
    return (original_angle + offset_deg) % 360

def apply_antenna_gain_adjustment(rssi_values, tx_angles, antenna_pattern, resolution):
    adjusted_rssi = []
    for i, rssi in enumerate(rssi_values):
        angle = int(round(tx_angles[i])) % 360
        idx = int(angle // resolution)
        gain = antenna_pattern[idx % antenna_pattern.shape[0], antenna_pattern.shape[1] // 2]
        adjusted_rssi.append(rssi + gain)
    return np.array(adjusted_rssi)

def augment_df(df, antenna_json, angle_offsets=[0], pos_noise_std=2.0, rssi_noise_std=1.5):
    gain_db, res = load_antenna_pattern(antenna_json)
    augmented = []
    for offset in angle_offsets:
        df_aug = df.copy()
        df_aug.iloc[:, 3] = apply_angular_offset(df_aug.iloc[:, 3], offset)
        df_aug.iloc[:, 0:2] += np.random.normal(0, pos_noise_std, size=(len(df_aug), 2))
        for col in range(5, 10):
            df_aug.iloc[:, col] += np.random.normal(0, rssi_noise_std, size=len(df_aug))
        df_aug.iloc[:, 5:10] = apply_antenna_gain_adjustment(df_aug.iloc[:, 5:10].values,
                                                             df_aug.iloc[:, 3].values,
                                                             gain_db, res)
        augmented.append(df_aug)
    return pd.concat(augmented, ignore_index=True)

def train_and_compare_split_aug(orig_file: str, antenna_json: str):
    df = pd.read_csv(orig_file) if orig_file.endswith(".csv") else pd.read_excel(orig_file)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    augmented_train_df = augment_df(train_df, antenna_json)
    combined_train_df = pd.concat([train_df, augmented_train_df], ignore_index=True)

    def compute_relative_angles(tx_pos, rx_positions):
        vectors = rx_positions - tx_pos
        horizontal_vectors = vectors[:, :2]
        norms = np.linalg.norm(horizontal_vectors, axis=1)
        norms[norms == 0] = 1e-6
        directions = horizontal_vectors / norms[:, None]
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        return np.degrees(angles) % 360

    receivers = np.array([
        [0, 0, 0],
        [36, 633, 2],
        [546, 596, 4],
        [297, 326, 6],
        [633, 70, 8]
    ])

    def prepare_data(df):
        tx_positions = df.iloc[:, 0:3].values
        rssi = df.iloc[:, 5:10].values
        azimuth_classes = (df.iloc[:, 3] // 10).astype(int).values
        angle_features = np.array([
            compute_relative_angles(tx, receivers) for tx in tx_positions
        ])
        X = np.hstack([rssi, angle_features])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, azimuth_classes

    X_train_orig, y_train_orig = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)

    clf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = clf_orig.predict(X_test)
    diff_orig = np.abs(y_pred_orig - y_test)

    X_train_aug, y_train_aug = prepare_data(combined_train_df)
    clf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_aug.fit(X_train_aug, y_train_aug)
    y_pred_aug = clf_aug.predict(X_test)
    diff_aug = np.abs(y_pred_aug - y_test)

    bins = np.arange(0, 10)
    x = np.arange(len(bins))
    counts_orig = np.array([(diff_orig == i).sum() for i in bins]) / len(diff_orig)
    counts_aug = np.array([(diff_aug == i).sum() for i in bins]) / len(diff_aug)

    bar_width = 0.35
    xtick_labels = [f"{i} class" for i in bins]

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width/2, counts_orig, width=bar_width, color="salmon", edgecolor="black", label="Original Train")
    plt.bar(x + bar_width/2, counts_aug, width=bar_width, color="mediumseagreen", edgecolor="black", label="Augmented Train")

    for i in range(len(bins)):
        plt.text(x[i] - bar_width/2, counts_orig[i] + 0.005, f"{counts_orig[i]*100:.1f}%", ha='center', fontsize=8)
        plt.text(x[i] + bar_width/2, counts_aug[i] + 0.005, f"{counts_aug[i]*100:.1f}%", ha='center', fontsize=8)

    plt.xticks(x, xtick_labels)
    plt.ylim(0, 1.0)
    plt.xlabel("Category Offset (|Predicted Class - True Class|)")
    plt.ylabel("Frequency")
    plt.title("Azimuth Angle Classification Error Comparison")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    acc0_orig = np.mean(diff_orig == 0) * 100
    acc1_orig = np.mean(diff_orig <= 1) * 100
    acc2_orig = np.mean(diff_orig <= 2) * 100
    acc0_aug = np.mean(diff_aug == 0) * 100
    acc1_aug = np.mean(diff_aug <= 1) * 100
    acc2_aug = np.mean(diff_aug <= 2) * 100

    print("📊 Accuracy on Unseen Original Test Set (Comparison):")
    print(f"🟥 Original Training - Exact match (0 class offset): {acc0_orig:.2f}%")
    print(f"🟥 Within ±0 class (max ±10°): {acc1_orig:.2f}%")
    print(f"🟥 Within ±2 classes (max ±30°): {acc2_orig:.2f}%")
    print(f"🟩 Augmented Training - Exact match (0 class offset): {acc0_aug:.2f}%")
    print(f"🟩 Within ±0 class (max ±10°): {acc1_aug:.2f}%")
    print(f"🟩 Within ±2 classes (max ±30°): {acc2_aug:.2f}%")

# 使用时替换路径
train_and_compare_split_aug(
    orig_file=r"d:\desktop\毕设材料\角度\angledata32303.csv",
    antenna_json=r"d:\desktop\毕设材料\角度\3GPP_sectorized_v_pol_65_hpbw.json"
)


# 现在输入一个代码，用他自己数据增强之前的和之后的结果进行对比，看看数据增强是否有效。
# 这个代码是用来对比数据增强前后的分类效果的
# 主要是通过对比分类器在原始数据和增强数据上的分类准确率来评估数据增强的效果。
# 具体步骤如下：
# 1. 加载原始数据和增强数据。
# 2. 对数据进行预处理，包括计算相对角度、标准化等。
# 3. 使用随机森林分类器对原始数据和增强数据进行训练和预测。
# 4. 计算分类器在原始数据和增强数据上的准确率，并输出结果。注意，他们的测试集是一样的。
# 5. 绘制分类器在原始数据和增强数据上的分类准确率对比图。
# 6. 输出分类器在原始数据和增强数据上的分类准确率对比结果。
#注意：我们的测试集不进行增强，只是用来评估分类器的性能。增强数据集是用来训练分类器的。
# 这样可以更好地评估数据增强的效果。防止增强过拟合。