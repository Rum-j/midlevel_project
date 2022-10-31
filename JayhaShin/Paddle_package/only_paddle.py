import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
parent = os.path.abspath('.')
sys.path.insert(1, parent)
from _ocr_func import _list_files, _img_ocr_result, _ocr_result_process, _heatmap_2, _kmeanclustered


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

img = _kmeanclustered(img, 64)


""" paddleocr """
try :    from PADDLE.paddleocr import PaddleOCR
except : from Paddle_package.PADDLE.paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en') # `en`, `ch`, `en`, `fr`, `german`, `korean`, `japan`
result = ocr.ocr(img, cls=True)


""" result(ndarray) --> ocr_result(dataframe) """
coor, value = map(list, zip(*result))
tl, _, br, _ = map(list, zip(*coor))
text, conf = map(list, zip(*value))
tl_x, tl_y = map(list, zip(*tl))
br_x, br_y = map(list, zip(*br))
ocr_result = pd.DataFrame(data=[], columns=["left", "top", "width", "height", "conf", "text"])
ocr_result["left"] = tl_x
ocr_result["top"] = tl_y
ocr_result["width"] = list(np.subtract(br_x, tl_x))
ocr_result["height"] = list(np.subtract(br_y, tl_y))
ocr_result["conf"] = conf
ocr_result["text"] = text

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
# print(z_grid.shape)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()
# plt.show()      # optional
plt.savefig(RESULT_DIR+"result_paddle.jpg")
