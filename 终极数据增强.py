import pandas as pd
import numpy as np
import json

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

def augment_data_advanced(input_path, output_path, antenna_json, angle_offsets=[3, -3, 7], pos_noise_std=2.0, rssi_noise_std=1.5):
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)
    gain_db, res = load_antenna_pattern(antenna_json)

    original_data = df.copy()
    augmented_data = []

    # 位置扰动
    pos_aug = original_data.copy()
    pos_aug.iloc[:, 0:2] += np.random.normal(0, pos_noise_std, size=(len(pos_aug), 2))
    augmented_data.append(pos_aug)

    # RSSI扰动
    rssi_aug = original_data.copy()
    for col in range(5, 10):
        rssi_aug.iloc[:, col] += np.random.normal(0, rssi_noise_std, size=len(rssi_aug))
    augmented_data.append(rssi_aug)

    # 角度扰动 + 位置 + RSSI + 天线增益
    for offset in angle_offsets:
        df_aug = original_data.copy()
        df_aug.iloc[:, 2] = apply_angular_offset(df_aug.iloc[:, 2], offset)
        df_aug.iloc[:, 0:2] += np.random.normal(0, pos_noise_std, size=(len(df_aug), 2))
        for col in range(5, 10):
            df_aug.iloc[:, col] += np.random.normal(0, rssi_noise_std, size=len(df_aug))
        df_aug.iloc[:, 5:10] = apply_antenna_gain_adjustment(df_aug.iloc[:, 5:10].values,
                                                             df_aug.iloc[:, 2].values,
                                                             gain_db, res)
        augmented_data.append(df_aug)

    all_data = pd.concat([original_data] + augmented_data, ignore_index=True)
    if output_path.endswith(".xlsx"):
        all_data.to_excel(output_path, index=False,header=False)
    else:
        all_data.to_csv(output_path, index=False, header=False)

    print(f"✅ 增强数据保存至 {output_path}，总样本数: {len(all_data)}")

# 示例运行：
augment_data_advanced(
    input_path=r"D:\desktop\毕设材料\角度\angledata32303.csv",
    output_path=r"D:\desktop\毕设材料\角度\angledata32303super.csv",
    antenna_json=r"D:\desktop\毕设材料\角度\3GPP_sectorized_v_pol_65_hpbw.json",
    angle_offsets=[3, -3, 7],  # 可自定义
    pos_noise_std=2.0,
    rssi_noise_std=1.5
)
