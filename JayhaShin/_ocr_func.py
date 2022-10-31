import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


""" load image """
def _list_files(dir='./input/'):
    import os
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files



# type result and draw bbox over image
def _img_ocr_result(img, ocr_result, FONT_SIZE=10):
    import cv2
    for _, row in ocr_result.iterrows():
        x, y, w, h = row["left"], row["top"], row["width"], row["height"]
        plt.text(x, (y - 10), row["text"], fontsize=FONT_SIZE, color="red")
        cv2.rectangle(img,(int(x), int(y)),(int(x) + int(w), int(y) + int(h)),(255, 0, 0),1 )
    return img


# K-means Clustering
def _kmeanclustered(img, KNUM=2):
    img_2 = img.copy().reshape(-1, 3)
    img_2 = np.float32(img_2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = KNUM
    ret, label, center = cv2.kmeans(img_2, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS )
    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((img.shape))
    return img


def _grayscale3ch(img) :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 1ch
    img = np.stack((img,img,img), axis=2, out=None)  # 1ch -> 3ch
    return img


# radial heatmap
def _heatmap_2(img, ocr_result):
    from scipy.interpolate.rbf import Rbf  # radial basis functions
    if len(ocr_result) <= 2:
        # avoid [ValueError: zero-size array to reduction operation maximum which has no identity]
        ocr_result = pd.DataFrame(data=[], columns=["left", "top", "width", "height", "conf", "text"])
        ocr_result.loc[len(ocr_result)] = [0, 0, 0, 0, 0.0, "1"]
        ocr_result.loc[len(ocr_result)] = [0, 1, 0, 0, 0.0, "1"]
    x = ocr_result["left"] + (ocr_result["width"] // 2)
    y = ocr_result["top"] + (ocr_result["height"] // 2)
    z = ocr_result["text"].astype(np.int64)
    # https://stackoverflow.com/questions/51647590/2d-probability-distribution-with-rbf-and-scipy
    rbf_adj = Rbf(x, y, z, function="gaussian")
    dh, dw, _ = img.shape
    x_fine = np.linspace(0, dw, num=81)  # (start, stop, step)
    y_fine = np.linspace(0, dh, num=82)
    x_grid, y_grid = np.meshgrid(x_fine, y_fine)
    z_grid = rbf_adj(x_grid.ravel(), y_grid.ravel()).reshape(x_grid.shape)
    return x_fine, y_fine, z_grid



"""
Tesseract installer for Windows
https://github.com/UB-Mannheim/tesseract/wiki
Change the directory below
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

!sudo apt install tesseract-ocr
!pip install pytesseract
!pip install tesseract
!sudo apt-get install tesseract-ocr-eng         # optional
!sudo apt-get install tesseract-ocr-kor         # optional

oem psm 모드 설명
https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc
https://stackoverflow.com/questions/65196162/easyocr-not-recognizing-simple-numbers
txt = pytesseract.image_to_string(final_image, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
"""
# pytesseract
def _ocr_pytesseract(img):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    traindata = None  # 'kor' 'eng' 'kor+eng' None
    # psm = 6 (0-13)
    # oem = 3 (0-3)
    ocr_result = pytesseract.image_to_data(
        image=img,
        lang=traindata,
        config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789",
        output_type="data.frame",
    )
    ocr_result = ocr_result[["left", "top", "width", "height", "conf", "text"]].copy()
    return ocr_result




"""
EasyOCR
https://github.com/JaidedAI/EasyOCR/blob/master/README.md
https://www.jaided.ai/easyocr/documentation/

!pip install easyocr
!pip install git+git://github.com/jaidedai/easyocr.git

def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
            workers = 0, allowlist = None, blocklist = None, detail = 1,\
            rotation_info = None, paragraph = False, min_size = 20,\
            contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
            text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
            canvas_size = 2560, mag_ratio = 1.,\
            slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
            width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
            threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
            output_format='standard'):
"""
def _easyocr(IMGDIR_or_IMAGE):
    import easyocr
    reader = easyocr.Reader(["en", "ko"])  # ['en', 'ko']
    bounds = reader.readtext(
        IMGDIR_or_IMAGE,
        blocklist="aCcDdLlIiJjOoSsUu그미ㅇ이",  # I need number, not alphabet
        min_size=0,
        low_text=0.1,
        link_threshold=30.0,
        canvas_size=np.inf,  # Image bigger than this value will be resized down
        ycenter_ths=0.0,  # maximum vertical shift
        height_ths=0.0,  # maximum different height
        width_ths=0.0,  # maximum horizontal shift
        add_margin=0.0,
        x_ths=0.0,
        y_ths=0.0,
        # rotation_info=[-20, 0, 20],
    )  # optional (allowlist ='0123456789')
    ocr_result = pd.DataFrame(data=[], columns=["left", "top", "width", "height", "conf", "text"] )
    if len(bounds) != 0:
        # EacyOCR result to dataframe
        coordinates, ocr_result_text, ocr_result_conf = map(
            list, zip(*bounds)
        )  # Splitting nested list
        tl, _, br, _ = map(list, zip(*coordinates))
        tl_x, tl_y = map(list, zip(*tl))
        br_x, br_y = map(list, zip(*br))
        ocr_result["left"] = tl_x
        ocr_result["top"] = tl_y
        ocr_result["width"] = list(np.subtract(br_x, tl_x))
        ocr_result["height"] = list(np.subtract(br_y, tl_y))
        ocr_result["conf"] = ocr_result_conf
        ocr_result["text"] = ocr_result_text
    return ocr_result


def _ocr_result_process(ocr_result, UPPER_LIMIT, CONFIDENCE=0.0):
    if len(ocr_result) != 0:
        ocr_result = ocr_result.dropna(axis=0).reset_index(drop=True)
        ocr_result["text"] = ocr_result["text"].astype("string")
        for idx, str_item in enumerate(ocr_result["text"]):
            for char in str_item:
                if not char.isdigit():
                    str_item = str_item.replace(char, "")
                    ocr_result.loc[
                        idx, "text"
                    ] = str_item  # remove non-numeric char from DataFrame, such as comma
                if len(str_item) < 3:
                    ocr_result.loc[idx, "text"] = ""
        ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)
        for idx, str_item in enumerate(ocr_result["text"]):
            if ocr_result.loc[idx, "conf"] < CONFIDENCE:  # threshold
                ocr_result.loc[idx, "text"] = ""
                continue
            if int(str_item) > UPPER_LIMIT:
                ocr_result.loc[idx, "text"] = ""
                continue
            if int(str_item) < 1000:                    # option for KRW (change if USD or other)
                ocr_result.loc[idx, "text"] = ""
                continue
            if (str_item[-2] + str_item[-1]) != "00":
                ocr_result.loc[idx, "text"] = ""
        ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)
        ocr_result.loc[len(ocr_result)] = [0, 0, 0, 0, 0.0, "0"]   # "scipy.interpolate.rbf" if all results were the same, then there is nothing to interpolate, so add an extra value
    return ocr_result


''' performance(accuracy) check of 500 random integers image synthesized by 17 different fonts  '''
''' Groundtruth is INT_LIST  '''
def _synthesize_performance_input_image(how_many_number=500, how_many_font=17, font_size=10) :
    # from itertools import product
    # combination = list(product([1,2,3,4,5,6,7,8,9,0], repeat=5))        # kernel crash
    import random
    random.seed(0)
    INT_LIST = [int(random.randrange(0, 99999, 5)) for _ in range(how_many_number)]
    print("Combination of 5 random digits")
    print("\tSample:", INT_LIST[0] )
    print("\tCount:", len(INT_LIST) )
    import matplotlib.font_manager as fm
    FONT_LIST = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    FONT_LIST = FONT_LIST[:min(how_many_font,len(FONT_LIST))]      # Started this work in Colab and it had only 17 fonts.
    print("Font list")
    print("\tSample:", FONT_LIST[0])
    print("\tCount:", len(FONT_LIST))
    groundtruth = len(INT_LIST) * len(FONT_LIST)
    print("Groundtruth(counts):", groundtruth )
    from PIL import Image, ImageDraw, ImageFont
    x_MARGIN = 80
    y_MARGIN = 50
    img_PIL = Image.new('RGB', (len(FONT_LIST)*x_MARGIN, len(INT_LIST)*y_MARGIN), color = (255,255,255))
    d = ImageDraw.Draw(img_PIL)
    for int_idx, int_val in enumerate(INT_LIST):
        for font_idx, font_val in enumerate(FONT_LIST):
            d.text((font_idx*x_MARGIN,int_idx*y_MARGIN), str(int_val), font=ImageFont.truetype(font=font_val, size=font_size), fill=(0,0,0))
    # display(img_PIL)
    # img_PIL.save('performance_input(50_50).jpg')
    # print(type(img_PIL))    # <class 'PIL.Image.Image'>
    import numpy as np
    img_np = np.array(img_PIL)
    # print(type(img_np), img_np.shape)   # <class 'numpy.ndarray'> (25000, 850, 3)
    return INT_LIST, FONT_LIST, img_PIL, img_np
# img_PIL, img_np = _synthesize_performance_input_image()
def _performance_output(ocr_result, INT_LIST, FONT_LIST) :
    ocr_result = ocr_result.dropna(axis=0).reset_index(drop=True)
    for idx, str_item in enumerate(ocr_result["text"]):
        for char in str_item:
            if len(str(str_item)) >= 6:
                ocr_result.loc[idx, "text"] = ""
            if not char.isdigit():
                ocr_result.loc[idx, "text"] = ""
    ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)
    groundtruth = len(INT_LIST) * len(FONT_LIST)
    r1 = []
    for _, row in ocr_result.iterrows():
        if int(row['text']) in INT_LIST:      # remove wrong answer
            aa = str(row['left'])+"__"+str(row['text'])
            r1 = r1 + [aa]
    counts = len(set(r1))              # remove duplicate
    print("Correct OCR counts:", counts )
    print("Groundtruth(counts):", groundtruth )
    print("Accuracy", counts/groundtruth*100,"%")
    return counts
#
