# Module to detect words


import numpy as np
from PIL import ImageGrab
import time
import cv2
import pytesseract



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('Images/test.jpg')
# heightImg = 640
# widthImg  = 480
heightImg , widthImg , _ = img.shape 
img = cv2.resize(img, (widthImg, heightImg))
#print(img)
#cv2.imshow("input",img)
#cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cv2.imshow("after colour conversion", img)
#print(pytesseract.image_to_string(img))

#print(pytesseract.image_to_boxes(img))
data = pytesseract.image_to_data(img)
#print(data)
# data columns

# level   page_num        block_num       par_num line_num        word_num        left    top     width   height  conf    text
for x, idata in enumerate(data.splitlines()):
    if(x!=0):
        if(len(idata)==12):
            idata=idata.split()
            print(idata)
            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img, (x,y), (w+x,y+h), (0,0,255), 3)
            cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0),1)

cv2.imshow("input",img)
cv2.waitKey(0)
print('end')
cv2.destroyAllWindows()