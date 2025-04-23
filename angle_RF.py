import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. 加载数据
data = pd.read_csv(r"D:\desktop\毕设材料\angledata32303.csv")

# 2. 使用高斯混合模型识别并标记噪声
cleaned_data = data.copy()

# 对每个接收机的信号进行清洗
for i in range(5):
    col_idx = i + 5  # 假设接收机数据从第6列开始
    
    # 使用高斯混合模型来自动识别信号和噪声
    signal_values = data.iloc[:, col_idx].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(signal_values)
    
    # 确定哪个组件是信号，哪个是噪声
    if gmm.covariances_[0] > gmm.covariances_[1]:
        signal_idx, noise_idx = 0, 1
    else:
        signal_idx, noise_idx = 1, 0
    
    # 计算自适应噪声阈值
    noise_mean = gmm.means_[noise_idx][0]
    noise_std = np.sqrt(gmm.covariances_[noise_idx][0][0])
    threshold = noise_mean + 2 * noise_std
    
    print(f"接收机 {i+1} 的噪声阈值: {threshold:.2f} dB")
    
    # 创建一个掩码，标记所有噪声点
    mask = data.iloc[:, col_idx] <= threshold
    
    # 将噪声点设为NaN
    cleaned_data.loc[mask, cleaned_data.columns[col_idx]] = np.nan

# 3. 计算每行非NaN值的数量(有效信号数)
cleaned_data['valid_signals'] = cleaned_data.iloc[:, 5:10].notna().sum(axis=1)
print("有效信号数量统计:")
print(cleaned_data['valid_signals'].value_counts())

# 4. 仅保留至少有3个有效信号的行
min_valid_signals = 3
filtered_data = cleaned_data[cleaned_data['valid_signals'] >= min_valid_signals]
print(f"原始数据行数: {len(data)}")
print(f"至少有{min_valid_signals}个有效信号的行数: {len(filtered_data)}")
print(f"保留数据比例: {len(filtered_data)/len(data)*100:.2f}%")

# 5. 为缺失值（NaN）填充一个特殊值
# 对于信号缺失，我们使用每个接收机的噪声阈值作为填充值
X = filtered_data.iloc[:, 5:10].copy()

# 使用每列的噪声阈值进行填充
for i in range(5):
    col_idx = i + 5
    # 获取该列的噪声阈值
    mask = data.iloc[:, col_idx] <= threshold
    noise_value = threshold - 5  # 比噪声阈值稍低一点
    # 使用噪声阈值填充NaN
    X.iloc[:, i].fillna(noise_value, inplace=True)

# 提取角度标签
angles = filtered_data.iloc[:, 3].values  # 角度值
angle_classes = (angles / 5).astype(int)  # 转换为类别索引(0-71)

# 6. 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, angle_classes, test_size=0.2, random_state=42
)

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 8. 训练随机森林分类器
print("训练随机森林分类器...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

clf.fit(X_train, y_train)

# 9. 模型评估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"分类准确率: {accuracy:.4f}")

# 10. 为所有角度(0-355度，步长5度)进行预测
all_angles = np.arange(0, 360, 5)
angle_results = {}

# 创建一个角度类别到数据索引的映射
angle_to_indices = {}
for i, angle in enumerate(angles):
    angle_class = int(angle / 5)
    if angle_class not in angle_to_indices:
        angle_to_indices[angle_class] = []
    angle_to_indices[angle_class].append(i)

# 对每个角度进行预测
for angle in all_angles:
    angle_class = angle // 5
    
    if angle_class in angle_to_indices and angle_to_indices[angle_class]:
        # 随机选择一个匹配的数据点
        sample_idx = np.random.choice(angle_to_indices[angle_class])
        sample_features = X_scaled[sample_idx]
        
        # 预测概率
        proba = clf.predict_proba([sample_features])[0]
        top_3_classes = np.argsort(proba)[-3:][::-1]
        top_3_angles = top_3_classes * 5
        top_3_probs = proba[top_3_classes]
        
        angle_results[angle] = {
            'top_predictions': top_3_angles,
            'probabilities': top_3_probs,
            'valid_signals': filtered_data.iloc[sample_idx]['valid_signals']
        }
    else:
        print(f"没有找到角度为 {angle}° 的样本")

# 11. 创建一个极坐标图显示所有角度的预测结果
plt.figure(figsize=(12, 12))
ax = plt.subplot(111, projection='polar')

# 绘制所有角度的真实值和预测值
for angle in all_angles:
    if angle in angle_results:
        true_angle_rad = np.radians(angle)
        
        # 绘制真实角度点（蓝色）
        ax.scatter(true_angle_rad, 1, color='blue', s=30, alpha=0.7)
        
        # 获取顶部预测
        pred_angle = angle_results[angle]['top_predictions'][0]
        pred_angle_rad = np.radians(pred_angle)
        
        # 绘制预测点（红色）
        ax.scatter(pred_angle_rad, 0.8, color='red', s=20, alpha=0.7)
        
        # 连接真实角度和预测角度
        ax.plot([true_angle_rad, pred_angle_rad], [1, 0.8], 'k-', alpha=0.2)

# 设置角度标签
ax.set_thetagrids(np.arange(0, 360, 45))
ax.set_title('Prediction Results for All Angles (0-355 degrees)')

# 添加图例
ax.scatter([], [], color='blue', s=30, label='True Angle')
ax.scatter([], [], color='red', s=20, label='Predicted Angle')
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 12. 创建预测误差图 - 展示预测误差随角度变化的模式
error_data = []
valid_signals_data = []

for angle, result in angle_results.items():
    pred_angle = result['top_predictions'][0]
    valid_signals = result['valid_signals']
    error = min(abs(angle - pred_angle), 360 - abs(angle - pred_angle))
    error_data.append((angle, error))
    valid_signals_data.append((angle, valid_signals))

# 按角度排序
error_data.sort(key=lambda x: x[0])
valid_signals_data.sort(key=lambda x: x[0])

angles_list = [x[0] for x in error_data]
errors_list = [x[1] for x in error_data]
valid_signals_list = [x[1] for x in valid_signals_data]

# 创建双Y轴图表
fig, ax1 = plt.subplots(figsize=(12, 6))

# 第一个Y轴：预测误差
color = 'tab:blue'
ax1.set_xlabel('Angle (degrees)')
ax1.set_ylabel('Prediction Error (degrees)', color=color)
ax1.bar(angles_list, errors_list, width=4, color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(np.arange(0, 360, 45))
ax1.grid(True, alpha=0.3)

# 第二个Y轴：有效信号数
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Valid Signals Count', color=color)
ax2.plot(angles_list, valid_signals_list, color=color, marker='o', linestyle='-', markersize=3)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(min_valid_signals-0.5, 5.5)  # 设置Y轴范围从最小有效信号数到5

plt.title('Prediction Error and Valid Signals Count by Angle')
plt.tight_layout()
plt.show()

# 创建真实角度与预测角度的散点图
plt.figure(figsize=(10, 10))

# 将类别标签转换为实际角度值
y_true_angles = y_test * 5   # 真实角度 = 类别索引 * 5
y_pred_angles = y_pred * 5   # 预测角度 = 类别索引 * 5

# 绘制散点图
plt.scatter(y_true_angles, y_pred_angles, alpha=0.5)
plt.plot([0, 360], [0, 360], 'r--', label='Perfect Prediction')
plt.xlabel('True Angle (degrees)')
plt.ylabel('Predicted Angle (degrees)')
plt.title('Predicted vs True Angles')
plt.xlim(0, 360)
plt.ylim(0, 360)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# 13. 计算整体误差统计
errors = np.array(errors_list)
mean_error = np.mean(errors)
median_error = np.median(errors)
max_error = np.max(errors)

print(f"平均预测误差: {mean_error:.2f}°")
print(f"中位数预测误差: {median_error:.2f}°")
print(f"最大预测误差: {max_error:.2f}°")

# 14. 分析不同有效信号数量对预测准确性的影响
valid_signals_values = np.unique(valid_signals_list)
error_by_signal_count = {}

for signal_count in valid_signals_values:
    mask = np.array(valid_signals_list) == signal_count
    errors_for_count = np.array(errors_list)[mask]
    error_by_signal_count[signal_count] = {
        'mean_error': np.mean(errors_for_count),
        'median_error': np.median(errors_for_count),
        'count': len(errors_for_count)
    }

print("\n不同有效信号数对预测准确性的影响:")
for signal_count, stats in error_by_signal_count.items():
    print(f"有效信号数 = {signal_count}: 平均误差 = {stats['mean_error']:.2f}°, 中位数误差 = {stats['median_error']:.2f}°, 样本数 = {stats['count']}")