# =============================================================
# This file conftains the simulation of ISP pipeline
#
# Weihang Yuan 2019
#
# Report bugs/suggestions:
# yuanweihang@xiaomi.com
# =============================================================

import cv2
import numpy as np
import isp_utility as utility
from scipy import signal        # convolutions
import os
import scipy.io as sco
import skimage
import math
import random
import rawpy

def WB_Mask(img, pattern, fr_now, fb_now, fg_now=1.0):
    wb_mask = np.ones(img.shape) * fg_now
    if  pattern == 'RGGB':
        wb_mask[0::2, 0::2] = fr_now
        wb_mask[1::2, 1::2] = fb_now
    elif  pattern == 'BGGR':
        wb_mask[1::2, 1::2] = fr_now
        wb_mask[0::2, 0::2] = fb_now
    elif  pattern == 'GRBG':
        wb_mask[0::2, 1::2] = fr_now
        wb_mask[1::2, 0::2] = fb_now
    elif  pattern == 'GBRG':
        wb_mask[1::2, 0::2] = fr_now
        wb_mask[0::2, 1::2] = fb_now
    return wb_mask

def nonlinear_masking(data, strength_multiplier=1.0, gaussian_kernel_size=[5, 5], gaussian_sigma=1.0, clip_range=[0, 255]):
    # Objective: improves the overall tone of the image
    # Inputs:
    #   strength_multiplier: >0. The higher the more aggressing tone mapping
    #   gaussian_kernel_size: kernel size for calculating the mask image
    #   gaussian_sigma: spread of the gaussian kernel for calculating the
    #                   mask image
    #
    # Source:
    # N. Moroney, “Local color correction using non-linear masking”,
    # Proc. IS&T/SID 8th Color Imaging Conference, pp. 108-111, (2000)
    #
    # Note, Slight changes is carried by mushfiqul alam, specifically
    # introducing the strength_multiplier

    print("----------------------------------------------------")
    print("Running tone mapping by non linear masking...")

    # convert to gray image
    # data = data.astype(np.float32)
    if (np.ndim(data) == 3):
        gray_image = utility.color_conversion(data).rgb2gray()
    else:
        gray_image = data

    # gaussian blur the gray image
    gaussian_kernel = utility.create_filter().gaussian(gaussian_kernel_size, gaussian_sigma)

    # the mask image:   (1) blur
    #                   (2) bring within range 0 to 1
    #                   (3) multiply with strength_multiplier
    mask = signal.convolve2d(gray_image, gaussian_kernel, mode="same", boundary="symm")
    mask = strength_multiplier * mask / clip_range[1]

    # calculate the alpha image
    temp = np.power(0.5, mask)
    if (np.ndim(data) == 3):
        width, height = utility.helpers(data).get_width_height()
        alpha = np.empty((height, width, 3), dtype=np.float32)
        alpha[:, :, 0] = temp
        alpha[:, :, 1] = temp
        alpha[:, :, 2] = temp
    else:
        alpha = temp

    # output
    return np.clip(clip_range[1] * np.power(data/clip_range[1], alpha), clip_range[0], clip_range[1])

def purple_fringe_removal(data, nsr_threshold=90.0, cr_threshold=128.0, clip_range=[0, 255]):
    # --------------------------------------------------------------
    # nsr_threshold: near saturated region threshold (in percentage)
    # cr_threshold: candidate region threshold
    # --------------------------------------------------------------
    width, height,_ = data.shape
    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]

    ## Detection of purple fringe
    # near saturated region detection
    nsr_threshold = clip_range[1] * nsr_threshold / 100
    temp = (r + g + b) / 3
    temp = np.asarray(temp)
    mask = temp > nsr_threshold
    nsr = np.zeros((width, height), dtype=np.int)
    nsr[mask] = 1

    # candidate region detection
    temp = r - b
    temp1 = b - g
    temp = np.asarray(temp)
    temp1 = np.asarray(temp1)
    mask = (temp < cr_threshold) & (temp1 > cr_threshold)
    cr = np.zeros((width, height), dtype=np.int)
    cr[mask] = 1

    # quantization
    qr = utility.helpers(r).nonuniform_quantization()
    qg = utility.helpers(g).nonuniform_quantization()
    qb = utility.helpers(b).nonuniform_quantization()

    g_qr = utility.edge_detection(qr).sobel(5, "gradient_magnitude")
    g_qg = utility.edge_detection(qg).sobel(5, "gradient_magnitude")
    g_qb = utility.edge_detection(qb).sobel(5, "gradient_magnitude")

    g_qr = np.asarray(g_qr)
    g_qg = np.asarray(g_qg)
    g_qb = np.asarray(g_qb)

    # bgm: binary gradient magnitude
    bgm = np.zeros((width, height), dtype=np.float32)
    mask = (g_qr != 0) | (g_qg != 0) | (g_qb != 0)
    bgm[mask] = 1

    fringe_map = np.multiply(np.multiply(nsr, cr), bgm)
    fring_map = np.asarray(fringe_map)
    mask = (fringe_map == 1)

    r1 = r
    g1 = g
    b1 = b
    r1[mask] = g1[mask] = b1[mask] = (r[mask] + g[mask] + b[mask]) / 3.

    output = np.empty(np.shape(data), dtype=np.float32)
    output[:, :, 0] = r1
    output[:, :, 1] = g1
    output[:, :, 2] = b1
    return np.float32(output)

def RGB_WB(img, fr_now, fb_now, fg_now):
    img = img.astype(np.float32)
    wb_mask = np.ones_like(img)
    wb_mask[:, : ,0] *= fr_now
    wb_mask[:, : ,1] *= fg_now
    wb_mask[:, : ,2] *= fb_now

    img *= wb_mask
    img = np.clip(img, 0, 255)
    return img

def global_contrast_normalization(X, s, lmda, epsilon):
    # replacement for the loop
    X_average = np.mean(X)
    print('Mean: ', X_average)
    X = X - X_average
    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lmda + np.mean(X**2))
    X = s * X / max(contrast, epsilon)
    X = np.clip(X, 0 ,1)
    X = X * 255
    # scipy can handle it
    return X.astype(np.uint8)

def linear_to_srgb(linear):
    x = linear.astype(np.float32) / 255.0
    x_origin = x.copy()
    x = x ** (1.0 / 2.4) * 1.055 - 0.055
    less = x_origin <= 0.0031308
    x[less] = x_origin[less] * 12.92
    x = np.clip(x, 0.0, 1.0)
    x = x * 255
    return x.astype(np.uint8)

def hdrp_srgb(input, matrix):
    # result = np.dot(input, matrix.T)
    R = input[:, :, 0]
    G = input[:, :, 1]
    B = input[:, :, 2]

    result = np.zeros_like(input, dtype=np.float32)
    result[:, :, 0] = R * matrix[0, 0] + G * matrix[0, 1] + B * matrix[0, 2]
    result[:, :, 1] = R * matrix[1, 0] + G * matrix[1, 1] + B * matrix[1, 2]
    result[:, :, 2] = R * matrix[2, 0] + G * matrix[2, 1] + B * matrix[2, 2]

    # return np.clip(result, 0, 255)
    return result

def hdrp_contrast(input, strength = 10, black_level = 0.0):
    scale = 0.8 + 0.3 / np.minimum(1.0, strength)
    inner_contrast = 3.141592 / (2.0 * scale)
    sin_contrast = np.sin(inner_contrast)
    slope = 255.0 / (2.0 * scale)
    constant = slope * sin_contrast
    factor = 3.141592 / (scale * 255.0)
    white_scale = 255.0 / (255.0 - black_level)

    output = factor * input
    output = slope * np.sin(output - inner_contrast) + constant
    output = (output - black_level) * white_scale
    return output

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    image = (image * 255.0).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image.astype(np.float32)

def hdrp_sharpen(input, strength):
    guassianed = cv2.GaussianBlur(input, (7, 7), sigmaX=0.0)
    differ = input - guassianed
    output = input + differ * strength
    output = np.clip(output, 0.0, 255.0)
    return output

def RAW2RGB(Iref, color_matrix=[], camera_wb=[], flag_list=[0, 1, 1, 1, 0, 3, 0, 1, 0, 1],
            gamma=1.5, wb_parameter=5, contrast_strength = 20.0, tonemap_strength = 1.5,):
    # 1. black level
    # 2. white balance
    # 3. Demosaick
    # 4. sRGB color correction (sensor RGB to linear sRGB)
    # 5. Dynamic range commpression (local tone mapping)
    # 6. Global tone mapping (gamma correction : linear sRGB to non-linear sRGB)
    # 7. Chromatic aberration correction
    # 8. Global contrast increase
    # 9. Sharping
    # 10. Auto white balance
    if flag_list[1]:
        if camera_wb == []:
            camera_wb = [1.687527, 1.0, 2.082254]
        wb_mask = WB_Mask(Iref, 'RGGB', camera_wb[0], camera_wb[2])
        Iref = Iref * wb_mask
        print("2 : white balance - wb:{}".format(camera_wb))

    if flag_list[2]:
        Iref = Iref * (2 ** 16)
        Iref = np.clip(Iref, 0.0, 2 ** 16 - 1)
        Iref = Iref.astype(np.uint16)
        out = cv2.cvtColor(Iref, cv2.COLOR_BAYER_BG2RGB)
        out = out.astype(np.float32)
        out = out / (2**16 - 1)
        # out_min = out.min()
        # out_max = out.max()
        # out = ((out - out_min) / (out_max - out_min))
        # out = out.astype(np.uint8)
        print("3 : demosaick")
    else:
        out = Iref

    if flag_list[3]:
        # color_matrix = dng.rgb_xyz_matrix[:, 0:3]
        if color_matrix == []:
            color_matrix = np.array([[0.24448879, 0.5810741, 0.17443706],
                                     [-0.00469436, 0.96864164, 0.03605274],
                                     [-0.00345951, -0.06517734, 1.0686369]], dtype=np.float32)
        out = hdrp_srgb(out, color_matrix)
        print("4 : sRGB color correction colormatrix:{}".format(color_matrix))
    else:
        out = out.astype(np.float32)

    out = np.clip(out, 0.0, 1.0)
    # change the value to [0, 255]
    if flag_list[5] == 1:
        tonemap = cv2.createTonemapDrago(gamma=gamma)
        out = tonemap.process(out.astype(np.float32)) * 255.0
        print("5 : tone mapping")
    elif flag_list[5] == 2:
        out = adjust_gamma(out, gamma)
        print("5 : tone mapping_2")
    elif flag_list[5] == 3:
        out = adjust_gamma(out, gamma)
        out = nonlinear_masking(out, strength_multiplier=tonemap_strength)
        print("5 : tone mapping_3")
    else:
        out *= 255.0

    if flag_list[6] == 1:
        out = purple_fringe_removal(out)
        print("6 : Chromatic aberration correction")

    if flag_list[7] == 1:
        out = hdrp_contrast(out, strength=contrast_strength)
        print("7 : global contrast increase")

    if flag_list[8]:
        out = hdrp_sharpen(out, strength=2.5)
        print("8 : sharpen")

    if flag_list[9] == 1:
        wb = cv2.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(wb_parameter)
        out = wb.balanceWhite(out.astype(np.uint8)).astype(np.float32)
        print("10 : auto white balance_grayworld")
    elif flag_list[9] == 2:
        wb = cv2.xphoto.createSimpleWB()
        wb.setP(wb_parameter)
        p = wb.getP()
        print('p:{}'.format(p))
        out = wb.balanceWhite(out.astype(np.uint8)).astype(np.float32)
        # out = RGB_WB(out, camera_wb[0], camera_wb[2], camera_wb[1])
        print("10 : auto white balance_SimpleWB")
    elif flag_list[9] == 2:
        wb = cv2.xphoto.LearningBasedWB()
        out = wb.balanceWhite(out.astype(np.uint8)).astype(np.float32)
        # out = RGB_WB(out, camera_wb[0], camera_wb[2], camera_wb[1])
        print("10 : auto white balance_learning_based")

    # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(outputname, out.astype(np.uint8))
    return out

def get_rgb2xyz(color_space="srgb", illuminant="d65"):
    # Objective: get the rgb2xyz matrix dependin on the output color space
    #            and the illuminant
    # Source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    if (color_space == "srgb"):
        if (illuminant == "d65"):
            return [[.4124564, .3575761, .1804375], \
                    [.2126729, .7151522, .0721750], \
                    [.0193339, .1191920, .9503041]]
        elif (illuminant == "d50"):
            return [[.4360747, .3850649, .1430804], \
                    [.2225045, .7168786, .0606169], \
                    [.0139322, .0971045, .7141733]]
        else:
            print("for now, color_space must be d65 or d50")
            return

    elif (color_space == "adobe-rgb-1998"):
        if (illuminant == "d65"):
            return [[.5767309, .1855540, .1881852], \
                    [.2973769, .6273491, .0752741], \
                    [.0270343, .0706872, .9911085]]
        elif (illuminant == "d50"):
            return [[.6097559, .2052401, .1492240], \
                    [.3111242, .6256560, .0632197], \
                    [.0194811, .0608902, .7448387]]
        else:
            print("for now, illuminant must be d65 or d50")
            return
    else:
        print("for now, color_space must be srgb or adobe-rgb-1998")
        return


class Reverse_ISP:
    def __init__(self, curve_path='./data/'):
        filename = os.path.join(curve_path, 'dorfCurvesInv.mat')
        inverseCRFs = sco.loadmat(filename)
        self.I_inv = inverseCRFs['invI']
        self.B_inv = inverseCRFs['invB']
        self.xyz2cam_all = np.array([[1.0234,-0.2969,-0.2266,-0.5625,1.6328,-0.0469,-0.0703,0.2188,0.6406]
                            ,[0.4913,-0.0541,-0.0202,-0.613,1.3513,0.2906,-0.1564,0.2151,0.7183]
                            ,[0.838,-0.263,-0.0639,-0.2887,1.0725,0.2496,-0.0627,0.1427,0.5438]
                            ,[0.6596,-0.2079,-0.0562,-0.4782,1.3016,0.1933,-0.097,0.1581,0.5181]])

    def ICRF_Map(self, img, index=0):
        invI_temp = self.I_inv[index, :]
        invB_temp = self.B_inv[index, :]
        [h, w, c] = img.shape
        bin = invI_temp.shape[0]
        Size = w*h*c
        tiny_bin = 9.7656e-04
        min_tiny_bin = 0.0039
        temp_img = np.copy(img)
        temp_img = np.reshape(temp_img, (Size))
        for i in range(Size):
            temp = temp_img[i]
            start_bin = 1
            if temp > min_tiny_bin:
                start_bin = np.int(math.floor(temp/tiny_bin - 1))
            for b in range(start_bin, bin):
                tempB = invB_temp[b]
                if tempB >= temp:
                    index = b
                    if index > 1:
                        comp1 = tempB - temp
                        comp2 = temp - invB_temp[index-1]
                        if comp2 < comp1:
                            index = index - 1
                    temp_img[i] = invI_temp[index]
                    break
        temp_img = np.reshape(temp_img, (h, w, c))
        return temp_img

    def RGB2XYZ(self, img):
        xyz = skimage.color.rgb2xyz(img)
        return xyz

    def XYZ2CAM(self, img, M_xyz2cam=0):
        if type(M_xyz2cam) is int:
            cam_index = np.random.random((1, 4))
            cam_index = cam_index / np.sum(cam_index)
            M_xyz2cam = (self.xyz2cam_all[0, :] * cam_index[0, 0] + \
                         self.xyz2cam_all[1, :] * cam_index[0, 1] + \
                         self.xyz2cam_all[2, :] * cam_index[0, 2] + \
                         self.xyz2cam_all[3, :] * cam_index[0, 3] \
                         )
            self.M_xyz2cam = M_xyz2cam

        M_xyz2cam = np.reshape(M_xyz2cam, (3, 3))
        M_xyz2cam = M_xyz2cam / np.tile(np.sum(M_xyz2cam, axis=1), [3, 1]).T
        cam = self.apply_cmatrix(img, M_xyz2cam)
        cam = np.clip(cam, 0, 1)
        return cam

    def apply_cmatrix(self, img, matrix):
        r = (matrix[0, 0] * img[:, :, 0] + matrix[0, 1] * img[:, :, 1]
             + matrix[0, 2] * img[:, :, 2])
        g = (matrix[1, 0] * img[:, :, 0] + matrix[1, 1] * img[:, :, 1]
             + matrix[1, 2] * img[:, :, 2])
        b = (matrix[2, 0] * img[:, :, 0] + matrix[2, 1] * img[:, :, 1]
             + matrix[2, 2] * img[:, :, 2])
        r = np.expand_dims(r, axis=2)
        g = np.expand_dims(g, axis=2)
        b = np.expand_dims(b, axis=2)
        results = np.concatenate((r, g, b), axis=2)
        return results

    def Reverse_to_demosaic(self, img_rgb):
        icrf_index = random.randint(0, 200)
        img_L = self.ICRF_Map(img_rgb, index=icrf_index)
        # Step 2 : from RGB to XYZ
        img_XYZ = self.RGB2XYZ(img_L)
        # Step 3: from XYZ to Cam
        img_Cam = self.XYZ2CAM(img_XYZ, M_xyz2cam=0)
        return img_Cam
