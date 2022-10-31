## Price Distribution
Picture the merchandise stand with smartphone camera and visualize it's price distrubution. 
Developed to instantly visualize if a stand is well displayed via prices. 
Beneficial to merchandising display personnel.
 - ex) Is profit-purposed product well displayed on the line of sight?
 - ex) Is lesser-product displayed on the outward?
 - ex) Are prices giving customer at least three choices? (Good, Better, Best)
<br />

### Example Results
![example](examples/result_price_groundtrugh_1.jpg)
![example](examples/result_ocr_12_plots.jpg)
<br /><br />



## Usages
Change "UPPER_LIMIT"(ex. UPPER_LIMIT = 2500) depanding on the price of your image.
<br /><br />

"PyTesseract"
 - https://pypi.org/project/pytesseract/
 - Tesseract installer for Windows: https://github.com/UB-Mannheim/tesseract/wiki
 - (Window) Change the directory below
 - pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
 - pip install pytesseract
 - pip install tesseract
<br /><br />

"EasyOCR"
 - https://github.com/JaidedAI/EasyOCR/blob/master/README.md
 - https://www.jaided.ai/easyocr/documentation/
 - pip install easyocr
 - pip install git+git://github.com/jaidedai/easyocr.git
<br /><br />

"PaddleOCR"
 - https://github.com/PaddlePaddle/PaddleOCR
 - https://github.com/PaddlePaddle/Paddle
 - pip install paddlepaddle shapely pyclipper scikit-image imgaug lmdb tqdm
<br /><br />

"Google GCP Vision API"
 - https://cloud.google.com/vision/docs/before-you-begin
 - pip install --upgrade google-cloud google-cloud-vision
 - JSON key required. (Not provided)
 - Exception: Image too large. Please limit to 75 megapixels.
<br /><br />
