import numpy as np

def calculate_statistics(values):
    mean_val = np.mean(values)
    sigma_val = np.std(values)
    m3s_val = abs(mean_val) + 3 * sigma_val
    m3s_nm = m3s_val * 1e3  # nm 단위 변환
    return m3s_nm
