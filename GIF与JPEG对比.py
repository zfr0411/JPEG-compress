# -*- coding:utf-8 -*-
import os

import numpy
import math
from scipy.fftpack import dct
from PIL import Image
import numpy as np


im1 = Image.open(u"动物卡通图片.jpg")
im2 = Image.open(u"动物卡通图片1.jpg")
im3 = Image.open(u'动物卡通图片-JPEG_Compressed.jpg')
im_arr1 = np.array(im1)
im_arr2 = np.array(im2)
im_arr3 = np.array(im3)
width1, height1 = im_arr1.shape[0], im_arr1.shape[1]
sumR, sumG, sumB = 0, 0, 0
sumR1, sumG1, sumB1 = 0, 0, 0
MSE, PSNR, temp = 0, 0, 0
MSE1, PSNR1, temp1 = 0, 0, 0
for i in range(width1):
    for j in range(height1):
        matrix1 = im_arr1[i][j]
        matrix2 = im_arr2[i][j]
        matrix3 = im_arr3[i][j]
        sumR = sumR+(int(matrix1[0])-int(matrix2[0]))*(int(matrix1[0])-int(matrix2[0]))
        sumG = sumG+(int(matrix1[1])-int(matrix2[1]))*(int(matrix1[1])-int(matrix2[1]))
        sumB = sumB+(int(matrix1[2])-int(matrix2[2]))*(int(matrix1[2])-int(matrix2[2]))

        sumR1 = sumR1 + (int(matrix1[0]) - int(matrix3[0])) * (int(matrix1[0]) - int(matrix3[0]))
        sumG1 = sumG1 + (int(matrix1[1]) - int(matrix3[1])) * (int(matrix1[1]) - int(matrix3[1]))
        sumB1 = sumB1 + (int(matrix1[2]) - int(matrix3[2])) * (int(matrix1[2]) - int(matrix3[2]))
MSE = (sumB+sumG+sumR)/width1/height1
MSE1 = (sumB1+sumG1+sumR1)/width1/height1
print ('the MSE of GIF is :', MSE)
print ('the MSE of JPEG is :', MSE1)
temp = 255/(math.sqrt(MSE))
temp1 = 255/(math.sqrt(MSE1))
PSNR = 20*math.log(temp, 10)
PSNR1 = 20*math.log(temp1, 10)
print ('the PSNR of GIF is :', PSNR)
print ('the PSNR of JPEG is :', PSNR1)

