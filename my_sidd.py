# ------------------------------------------------------------- #
# Paer: A High-Quality Denoising Dataset for Smartphone Cameras #
# ------------------------------------------------------------- #


from hdr_plus import process_stack
from scipy import stats
import scipy.optimize as sco
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

def MLE(imgs):
    b, h, w = imgs.shape
    std = np.zeros(shape = (h, w))
    mean = np.zeros(shape=(h, w))
    for j in range(h):
        for i in range(w):
            mean[j, i], std[j, i] = stats.norm.fit(imgs[:, j, i])
    return mean, std
def calc_CDF(X):
    sum = np.sum(X)
    cs = np.cumsum(X)
    cdf = cs / sum
    return cdf, cs
def calc_Dst(mean, std, Ae):
    x = np.linspace(0, 255, 256)
    gaussian_arr = np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))
    Ap = np.cumsum(gaussian_arr)
    # Ap = stats.norm.cdf(x)
    Wt = 1.0 / np.sqrt((1 - Ae) * Ae)
    dst = np.sum(Wt * np.square(Ap - Ae))
    return dst
def WLS(imgs, th = 3, wls = True):
    b, h, w = imgs.shape
    mean, std = MLE(imgs)
    dst_mean, dst_std = mean, std
    if(wls):
        # 筛选过分点
        for j in range(h):
            for i in range(w):
                tmp_list = list(imgs[:, j, i])
                # tmp_arr = imgs[:, j, i]
                high_th = np.min(mean[j, i] + 3 * std[j, i])
                low_th = np.max(mean[j, i] - 3 * std[j, i], 0)
                # tmp_arr[high_th ]
                for v in tmp_list:
                    if v < low_th or v > high_th:
                        tmp_list.remove(v)
                tmp_list = sorted(tmp_list)
                # 计算Ae
                Ae = calc_CDF(tmp_list)
                # 计算目标函数
                # calc_Dst(mean[j, i], std[j, i], Ae)
                cons = ({'type': 'ineq', 'fun': lambda M: -np.abs(M - mean[j, i]) + th},
                        {'type': 'ineq', 'fun': lambda S: -np.abs(S - mean[j, i]) + th},
                        )
                opt = sco.minimize(fun=calc_Dst, x0=[mean[j, i], std[j, i]], args=(Ae), constraints=cons, method = 'Nelder-Mead',options={'disp':True})
                dst_mean[j, i], dst_std[j, i] = opt['M'], opt['S']
    return dst_mean, dst_std

