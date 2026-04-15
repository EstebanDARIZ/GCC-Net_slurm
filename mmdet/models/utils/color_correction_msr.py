import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

class GaussianBlurConv():
    '''
    高斯滤波
    依据图像金字塔和高斯可分离滤波器思路加速

    Gaussian filtering.
    Uses image pyramid and separable Gaussian filtering to accelerate computation.
    '''
    def FilterGaussian(self, img, sigma):
        '''
        高斯分离卷积，按照x轴y轴拆分运算，再合并，加速运算

        Separable Gaussian convolution.
        The convolution is split along x-axis and y-axis for efficiency.
        '''
        # reject unreasonable demands
        if sigma > 300:
            sigma = 300
        # Compute kernel size (must be odd) (impair)
        kernel_size = round(sigma * 3 * 2 +1) | 1   
        # Most energy is concentrated within 3*sigma on each side, so the range is 6*sigma + 1 
        # (center pixel), it must be odd so |1 to froce the weak bit to 1 (odd number)
        
        # Create Gaussian kernel
        kernel = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma, ktype=cv2.CV_32F)
        
        # Temporary buffer
        temp = np.zeros_like(img)
        
        # Horizontal (x-axis) convolution
        for j in range(temp.shape[0]):
            for i in range(temp.shape[1]):
                # 内层循环展开
                v1 = v2 = v3 = 0
                for k in range(kernel_size):
                    # Align kernel center with pixel position
                    source = math.floor(i+ kernel_size/2 -k)        

                    # Mirror padding for boundaries
                    if source < 0:
                        source = source * -1            
                    if source > img.shape[1]:
                        source = math.floor(2 * (img.shape[1] - 1) - source)  
                    v1 += kernel[k] * img[j, source, 0]
                    if temp.shape[2] == 1: continue
                    v2 += kernel[k] * img[j, source, 1]
                    v3 += kernel[k] * img[j, source, 2]
                temp[j, i, 0] = v1
                if temp.shape[2] == 1: continue
                temp[j, i, 1] = v2
                temp[j, i, 2] = v3
        
        # Vertical (y-axis) convolution
        for i in range(img.shape[1]):         # height
            for j in range(img.shape[0]):
                v1 = v2 = v3 = 0
                for k in range(kernel_size):
                    source = math.floor(j + kernel_size/2 - k)

                    # Mirror padding vertically
                    if source < 0:
                        source = source * -1
                    if source > temp.shape[0]:
                        source = math.floor(2 * (img.shape[0] - 1) - source)   # 上下对称
                    v1 += kernel[k] * temp[source, i, 0]
                    if temp.shape[2] == 1: continue
                    v2 += kernel[k] * temp[source, i, 1]
                    v3 += kernel[k] * temp[source, i, 2]
                img[j, i, 0] = v1
                if img.shape[2] == 1: continue
                img[j, i, 1] = v2
                img[j, i, 2] = v3
        return img

    def FastFilter(self, img, sigma):
        '''
        快速滤波，按照图像金字塔，逐级降低图像分辨率，对应降低高斯核的sigma，
        当sigma转换成高斯核size小于10，再进行滤波，后逐级resize
        递归思路

        Fast Gaussian filtering using image pyramid.
        The image is recursively downsampled while reducing sigma.
        When the kernel size becomes small, standard Gaussian blur is applied.
        '''
        # reject unreasonable demands
        if sigma > 300:
            sigma = 300
        
        kernel_size = round(sigma * 3 * 2 + 1) | 1  
        
        # Stop recursion when kernel is too small
        if kernel_size < 3:
            return
        
        if kernel_size < 10:
            # img = self.FilterGaussian(img, sigma)
            # Use OpenCV Gaussian blur for small kernels
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)   # 官方函数
            return img
        else:
            # Stop if image resolution is too small
            if img.shape[1] < 2 or img.shape[0] < 2:
                return img
            
            sub_img = np.zeros_like(img)   

            # Downsample image 
            sub_img = cv2.pyrDown(img, sub_img)   #sigma ~= 1 mais en récursivité donc sigma_total = sigma_1 + sigma_2 + ...

            # Recursive filtering
            sub_img = self.FastFilter(sub_img, sigma/2.0)

            # Upsample back to original size      
            img = cv2.resize(sub_img, (img.shape[1], img.shape[0]))    
            return img        

    def __call__(self, x, sigma):
        x = self.FastFilter(x, sigma)
        return x

class Retinex(object):
    """
    Retinex-based image enhancement.

    SSR: Single Scale Retinex
    MSR: Multi Scale Retinex (better dynamic range & detail preservation)

    SSR: baseline
    MSR: keep the high fidelity and the dynamic range as well as compressing img
    MSRCR_GIMP: (Multi-Scale Retinex with Color Restoration)
      Adapt the dynamics of the colors according to the statistics of the first and second order.
      The use of the variance makes it possible to control the degree of saturation of the colors.
    """

    def __init__(self, model='MSR', sigma=[30, 150, 300], restore_factor=2.0, color_gain=10.0, gain=270.0, offset=128.0):
        self.model_list = ['SSR','MSR']
        if model in self.model_list:
            self.model = model
        else:
            raise ValueError
        self.sigma = sigma                       # Variance of the Gaussian kernel

        # Color restoration parameters
        self.restore_factor = restore_factor     # Controls the non-linearity of color restoration 
        self.color_gain = color_gain             # Controls the gain of color restoration

        # Intensity rescaling parameters
        self.gain = gain                         # Gain over the range of changes in image pixel values  
        self.offset = offset                     # Offset over the range of changes in image pixel values
        self.gaussian_conv = GaussianBlurConv()  # Instantiate the Gaussian operator

    def _SSR(self, img, sigma):
        """
        Single Scale Retinex.
        """
        filter_img = self.gaussian_conv(img, sigma)    # [h,w,c]
        retinex = np.log10(img) - np.log10(filter_img) # paper equation 
        return retinex

    def _MSR(self, img, simga):
        """
        Multi Scale Retinex.
        """
        retinex = np.zeros_like(img)
        # startime = time.time()
        for sig in simga:
            retinex += self._SSR(img, sig)
        retinex = retinex / float(len(self.sigma))
        # endtime = time.time()
        # print('test time of msr:', endtime - startime)
        return retinex

    def _colorRestoration(self, img, retinex):
        """
        Color restoration step of MSRCR.
        When computing log10() - log10(), aka the reflectance,  we lost the color information 
        """
        img_sum = np.sum(img, axis=2, keepdims=True)  # Summate at the channel level
        
        # Color restoration

        # Normalize the weight matrix and take its logarithm to obtain the color gain
        color_restoration = np.log10((img * self.restore_factor / img_sum) * 1.0 + 1.0)
        
        # Reassemble the subtracted Retinex image according to the weights and color gain
        img_merge = retinex * color_restoration * self.color_gain
        
        # Restore the image
        img_restore = img_merge * self.gain + self.offset
        return img_restore

    def _simplestColorBalance(self, img, low_clip, high_clip):
        """
        Simplest color balance algorithm.

        Args:
            img (ndarray): Input image.
            low_clip (float): Low clipping percentage.
            high_clip (float): High clipping percentage.
        Returns:
            img (ndarray): Color balanced image.
        
        Its principle is to clip a certain percentage of pixels from the low and high ends 
        of the histogram, and then stretch the remaining pixel values to the full range [0, 255].
        """
        total = img.shape[0] * img.shape[1]   # H x W 
        for i in range(img.shape[2]):  # for each chanel 
            # Compute histogram 
            unique, counts = np.unique(img[:, :, i], return_counts=True)  # Returns the position of the new list element in the old list and stores it as a list in s.
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
        return img

    def _MSRCR_GIMP(self, results):
        """
        Multi-Scale Retinex with Color Restoration (MSRCR) as used in GIMP.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            results (dict): Result dict with color corrected image.
          
        It's used for color correction in low-light image enhancement.
        """

        self.img = results['img']
        self.img = np.float32(self.img) + 1.0 # To avoid log(0), float32 for precision
        if self.model == 'SSR':
            self.retinex = self._SSR(self.img, self.sigma)
        elif self.model == 'MSR':
            self.retinex = self._MSR(self.img, self.sigma)
        # Apply color restoration and intensity rescaling
        self.img_restore = self._colorRestoration(self.img, self.retinex)
        results['img'] = self.img_restore

    def __call__(self, results):
        self._MSRCR_GIMP(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '{},sigma={},dynamic={}'.format(self.model, self.sigma)
        return repr_str

# single scale retinex
def SSR(img, sigma):
    gaussian_conv = GaussianBlurConv()
    filter_img = gaussian_conv(img, sigma)    # [h,w,c]
    retinex = np.log10(img) - np.log10(filter_img)
    return retinex

def MSR(img, sigma):
    retinex = np.zeros_like(img)
    # startime = time.time()
    for sig in sigma:
        retinex += SSR(img, sig)
    retinex = retinex / float(len(sigma))
    # endtime = time.time()
    # print('test time of msr:', endtime - startime)
    return retinex

def colorRestoration(img, retinex, restore_factor, color_gain, gain, offset):
    img_sum = np.sum(img, axis=2, keepdims=True)  # 在通道层面求和
    # 颜色恢复
    # 权重矩阵归一化 并求对数，得到颜色增益
    color_restoration = np.log10((img * restore_factor / img_sum) * 1.0 + 1.0)
    # 将Retinex做差后的图像，按照权重和颜色增益重新组合
    img_merge = retinex * color_restoration * color_gain
    # 恢复图像
    img_restore = img_merge * gain + offset
    return img_restore

def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)  # 返回新列表元素在旧列表中的位置，并以列表形式储存在s中
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img

class MultiRetinex(nn.Module):
    def __init__(self, in_channels, model='MSR', sigma=[30, 150, 300], restore_factor=2.0, color_gain=10.0, gain=128.0, offset=128.0):
        super(MultiRetinex, self).__init__()
        self.model_list = ['SSR','MSR']
        if model in self.model_list:
            self.model = model
        else:
            raise ValueError
        self.sigma = sigma        # 高斯核的方差
        # 颜色恢复
        self.restore_factor = restore_factor     # 控制颜色修复的非线性
        self.color_gain = color_gain             # 控制颜色修复增益
        # 图像恢复
        self.gain = gain           # 图像像素值改变范围的增益
        self.offset = offset       # 图像像素值改变范围的偏移量
        self.gaussian_conv = GaussianBlurConv() # 实例化高斯算子
        self.in_channels = in_channels
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of feature refine module."""
        self.conv_5_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0))
        self.conv_1_5 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2))
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initialize weights of feature refine module."""
        normal_init(self.conv_5_1, std=0.01)
        normal_init(self.conv_1_5, std=0.01)

    def forward(self, x):
        for i in range(len(x)):
            tmp = np.array(x[i].permute(1, 2, 0).detach().cpu())
            img = np.float32(tmp) + 1.0
            retinex = MSR(img, self.sigma)
            img_store = colorRestoration(img, retinex)
            tmp_feat = self.conv_5_1(img_store)

  

