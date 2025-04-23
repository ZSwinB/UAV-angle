import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 接收机位置
receivers = np.array([
    [0, 0, 2],
    [36, 633, 4],
    [546, 596, 6],
    [297, 326, 8],
    [633, 70, 10]
])

def compute_relative_angles(tx_pos, rx_positions):
    vectors = rx_positions - tx_pos
    horizontal_vectors = vectors[:, :2]
    norms = np.linalg.norm(horizontal_vectors, axis=1)
    norms[norms == 0] = 1e-6
    directions = horizontal_vectors / norms[:, None]
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    return np.degrees(angles) % 360

def classify_azimuth_with_error_hist(file_path: str):
    # 判断文件类型，支持 CSV 和 Excel
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        data = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")

    tx_positions = data.iloc[:, 0:3].values
    rssi = data.iloc[:, 5:10].values
    data['azimuth_class'] = (data.iloc[:, 3] // 10).astype(int)
    y = data['azimuth_class'].values

    angle_features = np.array([
        compute_relative_angles(tx, receivers) for tx in tx_positions
    ])

    X = np.hstack([rssi, angle_features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    pred_deg = y_pred * 10
    true_deg = y_test * 10
    angle_error = (pred_deg - true_deg + 180) % 360 - 180

    # 误差统计输出
    within_10 = np.mean(np.abs(angle_error) <= 10) * 100
    within_30 = np.mean(np.abs(angle_error) <= 30) * 100
    print(f"✅ 误差在 ±10° 以内的占比: {within_10:.2f}%")
    print(f"✅ 误差在 ±30° 以内的占比: {within_30:.2f}%")

    # 绘制误差直方图
    plt.figure(figsize=(10, 6))
    plt.hist(angle_error, bins=36, range=(-180, 180), color='skyblue', edgecolor='black')
    plt.title("Azimuth Prediction Error Histogram")
    plt.xlabel("Prediction Error (degrees)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return clf, scaler

# 示例：
# classify_azimuth_with_error_hist(r"D:\desktop\毕设材料\angledataset.xlsx")
# classify_azimuth_with_error_hist(r"D:\desktop\毕设材料\angledataset.csv")



# 示例调用方式：
# classify_azimuth_with_error_hist(r"D:\desktop\毕设材料\angledataset.xlsx")


# 示例用法：
# classify_azimuth_from_excel(r"D:\desktop\毕设材料\angledataset.xlsx")

if __name__ == "__main__":
    excel_path = r"D:\desktop\毕设材料\角度\angledata32303_augmented.csv"  # ← 替换为你自己的路径
    classify_azimuth_with_error_hist(excel_path)