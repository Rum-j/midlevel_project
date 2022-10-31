import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sys, os
parent = os.path.abspath('.')
sys.path.insert(1, parent)
from _ocr_func import _list_files, _img_ocr_result, _heatmap_2, _kmeanclustered, \
                        _easyocr, _ocr_result_process


UPPER_LIMIT = 2500

""" load a file from 'input' """
INPUT_DIR='./input/'
if not os.path.isdir(INPUT_DIR):
    os.mkdir(INPUT_DIR)
img_files, _, _ = _list_files(INPUT_DIR)
if len(img_files) == 0 :
    print('There isnt any image file inside the folder "input" ')
    import sys
    sys.exit("BREAK")
IMG_DIR = img_files[0]
original_img_ndarray = plt.imread(IMG_DIR)
dh, dw, _ = original_img_ndarray.shape
img = original_img_ndarray.copy()

RESULT_DIR='./__result__/'
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)
print("Result is saved in the folder "+RESULT_DIR)




""" preprocess """
img = _kmeanclustered(img, 64)


""" OCR """
ocr_result = _easyocr(img)
ocr_result = _ocr_result_process(ocr_result, UPPER_LIMIT)


""" visualize """
plt.figure(figsize=((dw/max(dw,dh)*8)*3,(dh/max(dw,dh)*8)) )
plt.subplot(1, 3, 1)
img_with_ocr = _img_ocr_result(img, ocr_result, FONT_SIZE=8)  # plt.text
plt.imshow(img_with_ocr)
plt.subplot(1, 3, 2)
plt.imshow(original_img_ndarray)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid, alpha=0.5)   # alpha
plt.subplot(1, 3, 3)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()
# plt.show()      # optional
plt.savefig(RESULT_DIR+"result_easyocr.jpg")
