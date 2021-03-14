import numpy as np
from PIL import ImageGrab
import time
import cv2
import pytesseract



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#
# img = cv2.imread('Images/test.jpg')
img = cv2.imread("C:/Users/Rishabh/Desktop/Rishabh/Projects/Document-Scanner/opencv_frame_1.png")
# heightImg = 640
# widthImg  = 480
heightImg , widthImg , _ = img.shape 
img = cv2.resize(img, (widthImg, heightImg))
print(img)
cv2.imshow("input",img)
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("after colour conversion", img)
print(pytesseract.image_to_string(img))

#print(pytesseract.image_to_boxes(img))
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b=b.split(' ')
    #print(b)
    x,y,w,h=int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(img, (x,heightImg- y), (w,heightImg- h), (0,0,255), 2)
    cv2.putText(img, b[0], (x, heightImg-y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0),1)

cv2.imshow("input",img)
cv2.waitKey(0)
cv2.destroyAllWindows()