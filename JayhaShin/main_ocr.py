import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from _ocr_func import _list_files, _img_ocr_result, _heatmap_2, _kmeanclustered, _ocr_pytesseract, \
                        _easyocr, _ocr_result_process, _grayscale3ch


import time
now = time.localtime()
print('Started: {}-{} {}:{}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min))   # KST = UTC+9
start=time.time()

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

""" prepare a folder 'result' """
RESULT_DIR='./__result__/'
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)
print("Result is saved in the folder "+RESULT_DIR)


""" OCR & visualize """
sizerate=8
plt.figure(figsize=((dw/max(dw,dh)*sizerate)*3,(dh/max(dw,dh)*sizerate*4)) )

plt.subplot(4, 3, 1)
plt.ylabel("Pytesseract", fontsize=14)
img = _grayscale3ch(img)    # preprocess
ocr_result = _ocr_pytesseract(img)
ocr_result = _ocr_result_process(ocr_result, UPPER_LIMIT)
img_with_ocr = _img_ocr_result(original_img_ndarray, ocr_result, FONT_SIZE=8)  # plt.text
plt.imshow(img_with_ocr)
plt.subplot(4, 3, 2)
plt.imshow(img)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid, alpha=0.5)   # alpha
plt.subplot(4, 3, 3)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()

img = original_img_ndarray.copy()

plt.subplot(4, 3, 4)
plt.ylabel("EasyOCR", fontsize=14)
ocr_result = _easyocr(img)
ocr_result = _ocr_result_process(ocr_result, UPPER_LIMIT)
img_with_ocr = _img_ocr_result(img, ocr_result, FONT_SIZE=8)  # plt.text
plt.imshow(img_with_ocr)
plt.subplot(4, 3, 5)
plt.imshow(img)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid, alpha=0.5)   # alpha
plt.subplot(4, 3, 6)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()

img = original_img_ndarray.copy()

plt.subplot(4, 3, 7)
plt.ylabel("PaddleOCR", fontsize=14)
from Paddle_package.PADDLE.paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en') # `en`, `ch`, `en`, `fr`, `german`, `korean`, `japan`
result = ocr.ocr(img, cls=True)
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
img_with_ocr = _img_ocr_result(img, ocr_result, FONT_SIZE=8)  # plt.text
plt.imshow(img_with_ocr)
plt.subplot(4, 3, 8)
plt.imshow(img)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid, alpha=0.5)   # alpha
plt.subplot(4, 3, 9)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()

img = original_img_ndarray.copy()

plt.subplot(4, 3, 10)
plt.ylabel("Google GCP Vision API", fontsize=14)
""" load a JSON file """
JSON_KEY_DIR='./vision_jsonkey/'
if not os.path.isdir(JSON_KEY_DIR):
    os.mkdir(JSON_KEY_DIR)
import os
json_files = []
for (dirpath, _, filenames) in os.walk(JSON_KEY_DIR):
    for file in filenames:
        filename, ext = os.path.splitext(file)
        ext = str.lower(ext)
        if ext == '.json' :
            json_files.append(os.path.join(dirpath, file))
if len(json_files) == 0 :
    print('There isnt any json file inside the folder "vision_jsonkey" ')
    import sys
    sys.exit("BREAK")
JSON_KEY_DIR = json_files[0]
def detect_text(IMG_DIR, JSON_KEY_DIR):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(JSON_KEY_DIR)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    with io.open(IMG_DIR, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    ''' texts(proto.marshal.collections.repeated.RepeatedComposite) --> ocr_result(pandas.core.frame.DataFrame) '''
    temp_text_list = []
    temp_ocr_result = []
    for text in texts:
        temp_text_list = temp_text_list + [text.description]
        vertices = ([(vertex.x,vertex.y) for vertex in text.bounding_poly.vertices])
        temp_ocr_result = temp_ocr_result + list(vertices)
    import numpy as np
    temp_ocr_result = np.reshape(temp_ocr_result,(len(temp_ocr_result)//4,8))
    tl_x, tl_y, _, _, br_x, br_y, _, _ = map(list, zip(*temp_ocr_result))
    import pandas as pd
    ocr_result = pd.DataFrame(data=[], columns=["left", "top", "width", "height", "conf", "text"])
    ocr_result["left"] = tl_x
    ocr_result["top"] = tl_y
    ocr_result["width"] = list(np.subtract(br_x, tl_x))
    ocr_result["height"] = list(np.subtract(br_y, tl_y))
    ocr_result["conf"] = 0.  # N/A in Google Vision
    ocr_result["text"] = temp_text_list
    return response, texts, ocr_result
response, texts, ocr_result = detect_text(IMG_DIR, JSON_KEY_DIR)
ocr_result = _ocr_result_process(ocr_result, UPPER_LIMIT)
img_with_ocr = _img_ocr_result(img, ocr_result, FONT_SIZE=8)  # plt.text
plt.imshow(img_with_ocr)
plt.subplot(4, 3, 11)
plt.imshow(img)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid, alpha=0.5)   # alpha
plt.subplot(4, 3, 12)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()


# plt.show()      # optional
plt.savefig(RESULT_DIR+"result_ocr_9.jpg")

print('Duration: {:.0f}m {:.0f}s'.format( (time.time()-start)//60, (time.time()-start)%60) )
# ~ 10min
