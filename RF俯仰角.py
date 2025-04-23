import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 接收机位置
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

def classify_elevation_with_confusion_and_stats(file_path: str):
    # 支持 Excel 和 CSV 文件
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        data = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")

    tx_positions = data.iloc[:, 0:3].values
    rssi = data.iloc[:, 5:10].values
    data['elevation_class'] = (data.iloc[:, 4] // 10).astype(int)
    y = data['elevation_class'].values

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

    # 统计误差（单位：度）
    pred_deg = y_pred * 10
    true_deg = y_test * 10
    angle_error = np.abs(pred_deg - true_deg)

    within_10 = np.mean(angle_error <= 10) * 100
    within_30 = np.mean(angle_error <= 30) * 100
    print(f"✅ 误差在 ±10° 以内的占比: {within_10:.2f}%")
    print(f"✅ 误差在 ±30° 以内的占比: {within_30:.2f}%")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(18))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(18))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Elevation Angle Classification - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return clf, scaler

# 示例调用方式：
# classify_elevation_with_confusion_and_stats(r"D:\desktop\毕设材料\angledataset.xlsx")
# classify_elevation_with_confusion_and_stats(r"D:\desktop\毕设材料\angledataset.csv")



# 示例调用（将路径换成你自己的 Excel 文件）
# classify_with_rssi_and_angles(r"D:\desktop\毕设材料\angledataset.xlsx")



# ✅ 主程序入口
if __name__ == "__main__":
    excel_path = r"D:\desktop\毕设材料\角度\fuyangjiaosuper.xlsx"  # ← 替换为你自己的路径
    classify_elevation_with_confusion_and_stats(excel_path)
