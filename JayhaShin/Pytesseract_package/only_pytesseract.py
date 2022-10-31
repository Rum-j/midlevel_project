import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import time
now = time.localtime()
print('Started: {}-{} {}:{}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min))   # KST = UTC+9
start=time.time()


import sys, os
parent = os.path.abspath('.')
sys.path.insert(1, parent)
from _ocr_func import _list_files, _img_ocr_result, _heatmap_2, _kmeanclustered, _ocr_pytesseract, \
                    _ocr_result_process, _grayscale3ch


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
# img = _kmeanclustered(img, 64)   # not effective
img = _grayscale3ch(img)


""" OCR """
ocr_result = _ocr_pytesseract(img)
ocr_result = _ocr_result_process(ocr_result, UPPER_LIMIT)

# print(ocr_result)

""" visualize """
sizerate=6
plt.figure(figsize=((dw/max(dw,dh)*sizerate)*3,(dh/max(dw,dh)*sizerate)) )
plt.subplot(1, 3, 1)
plt.ylabel("Pytesseract", fontsize=14)
img_with_ocr = _img_ocr_result(original_img_ndarray, ocr_result, FONT_SIZE=8)  # plt.text
plt.imshow(img_with_ocr)
plt.subplot(1, 3, 2)
plt.imshow(img)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid, alpha=0.5)   # alpha
plt.subplot(1, 3, 3)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()
# plt.show()      # optional
plt.savefig(RESULT_DIR+"result_pytesseract.jpg")

print('Duration: {:.0f}m {:.0f}s'.format( (time.time()-start)//60, (time.time()-start)%60) )
