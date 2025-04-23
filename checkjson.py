import json
import numpy as np
import matplotlib.pyplot as plt

def load_gain_from_json_combined(json_path):
    """
    支持 resolution、theta/phi 极化合并的方向图解析函数。
    返回：gain_db, azimuth_angles, elevation_angles
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def extract_complex_gain(key):
        resolution = data[key].get("resolution", 1)
        real = np.array(data[key]["real"])
        imag = np.array(data[key].get("imag", np.zeros_like(real)))
        gain = real**2 + imag**2
        return gain, resolution

    gain_theta, res_theta = extract_complex_gain("theta")

    if "phi" in data:
        gain_phi, res_phi = extract_complex_gain("phi")
        assert res_phi == res_theta, "theta/phi 分辨率不一致"
        gain_total = gain_theta + gain_phi
    else:
        gain_total = gain_theta

    gain_db = 10 * np.log10(np.clip(gain_total, 1e-9, None))

    num_az, num_el = gain_db.shape
    azimuths = np.linspace(0, (num_az - 1) * res_theta, num_az)
    elevations = np.linspace(0, (num_el - 1) * res_theta, num_el)

    return gain_db, azimuths, elevations

def plot_polar_gain(gain_db, angles_deg, gain_values, title="Radiation Pattern"):
    theta_rad = np.radians(angles_deg)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(theta_rad, gain_values, linewidth=2)
    ax.plot(np.linspace(0, 2 * np.pi, 360), 
        np.full(360, np.max(gain_db) - 3), 
        linestyle='--', color='gray', label='-3 dB Beamwidth')

    ax.set_title(title, va='bottom')
    ax.set_rticks([-20, -10, 0, 5])
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    json_path = r"D:\desktop\毕设材料\角度\3GPP_sectorized_v_pol_65_hpbw.json"  # ← 替换成你自己的路径
    gain_db, az_list, el_list = load_gain_from_json_combined(json_path)

    # Azimuth cut：俯仰角固定为中间值
    el_mid_idx = len(el_list) // 2
    plot_polar_gain(gain_db, az_list, gain_db[:, el_mid_idx], 
                    title=f"Azimuth Cut (Elevation = {el_list[el_mid_idx]:.1f}°)")

    # Elevation cut：方位角固定为0°
    plot_polar_gain(gain_db, el_list, gain_db[0, :],
                    title="Elevation Cut (Azimuth = 0°)")


