import pandas as pd

import numpy as np


def full_data_augmentation(data: pd.DataFrame,
                           n_augment: int = 3,
                           rssi_noise_std: float = 1.5,
                           angle_perturb: int = 5,
                           pos_noise_std_xy: float = 2.0,
                           pos_noise_std_z: float = 0.5) -> pd.DataFrame:
    """
    同时执行位置误差模拟 + RSSI扰动 + 方向扰动的数据增强过程
    参数:
    - data: 原始DataFrame
    - n_augment: 每个样本生成的增强数量
    - rssi_noise_std: RSSI噪声标准差
    - angle_perturb: 角度扰动范围（±度）
    - pos_noise_std_xy: X/Y位置误差标准差
    - pos_noise_std_z: Z位置误差标准差
    返回:
    - 增强后的完整数据集（原始 + 增强）
    """
    augmented_rows = []

    for _, row in data.iterrows():
        for _ in range(n_augment):
            new_row = row.copy()

            # 位置扰动
            x_noise, y_noise = np.random.normal(0, pos_noise_std_xy, size=2)
            z_noise = np.random.normal(0, pos_noise_std_z)
            new_row.iloc[0] += x_noise
            new_row.iloc[1] += y_noise
            new_row.iloc[2] += z_noise

            # 方向扰动
            azimuth = new_row.iloc[3] + np.random.choice([-angle_perturb, 0, angle_perturb])
            azimuth = azimuth % 360
            elevation = new_row.iloc[4] + np.random.choice([-angle_perturb, 0, angle_perturb])
            elevation = np.clip(elevation, 0, 180)
            new_row.iloc[3] = azimuth
            new_row.iloc[4] = elevation

            # RSSI扰动
            rssi = new_row.iloc[5:10].values
            rssi_aug = rssi + np.random.normal(0, rssi_noise_std, size=rssi.shape)
            new_row.iloc[5:10] = rssi_aug

            augmented_rows.append(new_row)

    # 构造新DataFrame
    augmented_df = pd.DataFrame(augmented_rows, columns=data.columns)
    full_df = pd.concat([data, augmented_df], ignore_index=True)

    return full_df






# 读取你的原始数据
#df = pd.read_excel(r"D:\desktop\毕设材料\角度\fuyangjiao.xlsx", header=None)
df = pd.read_excel(r"D:\desktop\毕设材料\角度\fuyangjiao.xlsx", header=None)


# 调用增强函数（每个样本扩 3 个）
augmented_df = full_data_augmentation(df, n_augment=3)

# 保存增强后的数据（可选）
#augmented_df.to_excel(r"D:\desktop\毕设材料\角度\fuyangjiao_augmented.xlsx", index=False)
augmented_df.to_excel(r"D:\desktop\毕设材料\角度\fuyangjiao_augmented.xlsx", index=False, header=None)
