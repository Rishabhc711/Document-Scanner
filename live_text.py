import numpy as np
from PIL import ImageGrab
import time
import cv2
import pytesseract



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

##############################################
##### Webcam and Screen Capture Example ######
##############################################
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
cam.set(3,640)
cam.set(4,480)

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()


def captureScreen(bbox=(300,300,1500,1000)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr
while True:
    timer = cv2.getTickCount()
    #_,img = cap.read()
    #img = captureScreen()
    #DETECTING CHARACTERES
    img=cv2.imread("opencv_frame_{}.png".format(img_counter-1))
    hImg, wImg,_ = img.shape
    print(pytesseract.image_to_string(img))
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        #print(b)
        b = b.split(' ')
        #print(b)
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)
        cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    #cv2.putText(img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,230,20), 2);
    cv2.imshow("Result",img)
    cv2.waitKey(1)

cv2.imshow('img', img)
cv2.waitKey(0)
print('hi')
cv2.destroyAllWindows()