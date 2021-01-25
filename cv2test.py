from  cv2 import cv2
import numpy as np
#import utlis
 
 
########################################################################
webCamFeed = True
pathImage = "DOCSCANNER\\Images\\20210114_230355.jpg"
cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 640
widthImg  = 480
########################################################################
 
#utlis.initializeTrackbars()
count=0
def nothing(x=0):
    pass
    #cv2.imshow("Canny Image",ImageCanny)

cv2.namedWindow("Threshold Parameters")
cv2.resizeWindow("Threshold Parameters",400,600)
cv2.createTrackbar("Threshold1","Threshold Parameters",60,255,nothing)
cv2.createTrackbar("Threshold2","Threshold Parameters",60,255,nothing)

def getBiggestcontour(contours):
    #contours, hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgcontour,contours, -1,"red",5)
    for contour in contours:
        area=cv2.contourArea(contour)
        if area > 2000:
            peri=cv2.arcLength(contour,True)
            shape=cv2.approxPolyDP(contour,0.02*peri,True) 
            #print(len(shape))
            if(len(shape)==4):
                print(len(shape))
                #cv2.drawContours(imgcontour,contours, -1,(255, 255,0),5)
                cv2.drawContours(imgcontour,contour, -1,"blue",5)
    return imgcontour
while True:
 
    #if webCamFeed:success, img = cap.read()
    #else:
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    '''
    cv2.imshow("Image1",img)
    cv2.imshow("BLank Image",imgBlank)
    cv2.imshow("Blurred Image",imgBlur)
    cv2.waitKey()
    '''
    #cv2.destroyAllWindows()
     # ADD GAUSSIAN BLUR
    #thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    #imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    th1=cv2.getTrackbarPos("Threshold1","Threshold Parameters")
    th2=cv2.getTrackbarPos("Threshold2","Threshold Parameters") 
    ImageCanny=cv2.Canny(imgBlur,th1,th2)
    cv2.imshow("Canny Image",ImageCanny)
    cv2.waitKey(0)
    print('hi')
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(ImageCanny, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    '''
    cv2.imshow("Dilated",imgDial)
    cv2.imshow("Eroded",imgThreshold)
    cv2.waitKey()
    '''
    
    imgcontour=img.copy()
     #imgcontour=getcontours(ImageCanny,imgcontour)
    contours, hierarchy =cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(imgcontour,contours, -1,(0, 255, 0), 20)
    #cv2.imshow("Threshold image", imgcontour)
      # contour is also destructive
    #img = cv2.cvtColor(imgcontour, cv2.COLOR_GRAY2BGR)  #add this line
    cv2.drawContours(imgcontour, contours, -1, (0,255,0), 10)  # I am expecting the contour lines to be green
    cv2.imshow("Contour Image With colours", imgcontour)
    cv2.waitKey(0)
    img_with_corners=img.copy()
    corners = cv2.goodFeaturesToTrack(imgBlur, 150, 0.2, 50)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_with_corners, (x, y), 5, (200, 0, 255), -1)
    cv2.imshow("Image With corners", img_with_corners)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        break

    
    
'''
cv2.imshow(imgBlank)
cv2.imshow(imgBlur)
cv2.imshow(imgDial)
cv2.imshow(imgThreshold)
'''
print(cv2.__version__)
print('hi')
cv2.destroyAllWindows()