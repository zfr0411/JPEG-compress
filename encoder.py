# -*- coding: utf-8 -*
import argparse
import os
import math
import numpy as np
import numpy
from utils import *
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree


def quantize(block, component):
    q = load_quantization_table(component)
    return (block / q).round().astype(np.int32)


def block_to_zigzag(block):
    PI = 3.141593
    """for i in range(len(image)):
        for j in range(image[0]):
            tmp = 0.0

            if i == 0:
                coefficient1 = math.sqrt(1.0 / image.width)
            else:
                coefficient1 = math.sqrt(2.0 / image.width)
            if j == 0:
                coefficient2 = math.sqrt(1.0 / image.width)
            else:
                coefficient2 = math.sqrt(2.0 / image.width)
            for m in range(image[0]):
                for n in range(image[0]):
                    tmp += image[m][n] * math.cos((2 * m + 1) * PI * i / (2 * image[0])) * math.cos(
                        (2 * n + 1) * PI * j / (2 * image[0]))
            image[i][j] = round(coefficient1 * coefficient2 * tmp)
    return image
    """
    #以上代码为二维DCT变换，由于有四重循环，运行较慢，使用内置包作为代替，实现效果一致
    return np.array([block[point] for point in zigzag_points(*block.shape)])


def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def run_length_encode(arr):
    # 确定顺序在哪里过早结束
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    #  symbol  (RUNLENGTH, SIZE) tuple
    symbols = []

    # 值是使用大小位的数组元素的二进制表示形式
    values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values


def write_to_file(filepath, dc, ac, blocks_count, tables):

    f = open(filepath, 'w')
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        # 16 bits 'table_size'
        f.write(uint_to_binstr(len(tables[table_name]), 16))

        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                # 4 bits  'category'
                # 4 bits 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)
            else:
                # 4 bits 'run_length'
                # 4 bits  'size'
                # 8 bits 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key[0], 4))
                f.write(uint_to_binstr(key[1], 4))
                f.write(uint_to_binstr(len(value), 8))
                f.write(value)

    # 32 bits 'blocks_count'
    f.write(uint_to_binstr(blocks_count, 32))

    for b in range(blocks_count):
        for c in range(3):
            category = bits_required(dc[b, c])
            symbols, values = run_length_encode(ac[b, :, c])

            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']

            f.write(dc_table[category])
            f.write(int_to_binstr(dc[b, c]))

            for i in range(len(symbols)):
                f.write(ac_table[tuple(symbols[i])])
                f.write(values[i])
    f.close()


def main():
    print("请在命令行运行此程序")
    #imageName = input("Choose Image you want encode : ")
    imageName = raw_input("Choose Image you want encode : ")
    im = Image.open(imageName)
    image = im.resize((800, 800))  # 将图片定义为固定尺寸，方便进行图片操作，将图片切成8*8的快
    ycbcr = image.convert('YCbCr')  # 一、颜色模式转化
    # 图像矩阵化
    image_arr = np.array(image)
    npmat = np.array(ycbcr, dtype=np.uint8)  # 获取图像的亮度、色度与饱和度
    width, height = image_arr.shape[0], image_arr.shape[1]
    rows, cols = npmat.shape[0], npmat.shape[1]
    blocks_count = rows / 8 * cols / 8

    """
    # 颜色转化，将RGB转换为YCbCr
    Y = numpy.ndarray(shape=(width, height), dtype=np.int16)
    Cb = numpy.ndarray(shape=(width, height), dtype=np.int16)
    Cr = numpy.ndarray(shape=(width, height), dtype=np.int16)
    for i in range(width):
        for j in range(height):
            Image_Matrix = image_arr[i][j]
            Y[i][j] = 0.299 * Image_Matrix[0] + 0.587 * Image_Matrix[1] + 0.114 * Image_Matrix[2]
            Cb[i][j] = -0.1687 * Image_Matrix[0] - 0.3313 * Image_Matrix[1] + 0.5 * Image_Matrix[2] + 128
            Cr[i][j] = 0.5 * Image_Matrix[0] - 0.418 * Image_Matrix[1] - 0.0813 * Image_Matrix[2] + 128

    # 色彩二度采样 将YCbCr转化为Y：U：V 4：2 ：0 形式,y不变，U，V分别变为原来的1/4
    half_u = numpy.ndarray(shape=(width, height / 2), dtype=np.int16)
    half_v = numpy.ndarray(shape=(width, height / 2), dtype=np.int16)
    qut_u = numpy.ndarray(shape=(width / 2, height / 2), dtype=np.int16)
    qut_V = numpy.ndarray(shape=(width / 2, height / 2), dtype=np.int16)
    for m in range(height / 2):
        half_u[:, m] = Cb[:, m * 2]  # 从第0行开始，隔一行出现Cb
        half_v[:, m] = Cr[:, m * 2 + 1]  # 从第1行开始，隔一行出现Cr

    for n in range(width / 2):
        qut_u[n, :] = half_u[n * 2, :]  # 隔一列出现CbCr       实现 4：2：0效果
        qut_V[n, :] = half_v[n * 2, :]

    # 采样后YUV图像
    YUV = numpy.ndarray(shape=(width, height, 3), dtype=np.int16)
    # 组成YUV
    for i in range(rows):
        for j in range(cols):
            YUV[i][j][0] = Y[i][j]  # Y不变
            if (j % 2 == 0) & (i % 2 == 0):
                YUV[i][j][1] = qut_u[i / 2, j / 2]  # U每隔四个点出现一次
            if (j % 2 == 0) & (i % 2 != 0):
                YUV[i][j][2] = qut_V[i / 2, j / 2]  # U每隔四个点出现一次
    """
    dc = np.empty((blocks_count, 3), dtype=np.int32)   #
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                block_index += 1
            except NameError:
                block_index = 0

            for k in range(3):
                block = npmat[i:i+8, j:j+8, k] - 128

                dct_matrix = dct_2d(block)
                quant_matrix = quantize(dct_matrix,
                                        'lum' if k == 0 else 'chrom')
                zz = block_to_zigzag(quant_matrix)

                dc[block_index, k] = zz[0]
                ac[block_index, :, k] = zz[1:]

    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    H_AC_Y = HuffmanTree(
            flatten(run_length_encode(ac[i, :, 0])[0]
                    for i in range(blocks_count)))
    H_AC_C = HuffmanTree(
            flatten(run_length_encode(ac[i, :, j])[0]
                    for i in range(blocks_count) for j in [1, 2]))

    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}

    Basename = os.path.basename(imageName)
    base = os.path.splitext(Basename)[0]
    write_to_file(base + ".txt", dc, ac, blocks_count, tables)

if __name__ == "__main__":
    main()
