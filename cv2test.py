#import necessary modules/ libraries
from  cv2 import cv2
import numpy as np
#import utlis
 
 
########################################################################
# For Live Webcan feed
webCamFeed = True

cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 640
widthImg  = 480
########################################################################
# Filepath of Image 
pathImage = "DOCSCANNER\\Images\\test1.jpeg" 
#utlis.initializeTrackbars()

count=0
def nothing(x=0):
    pass
    #cv2.imshow("Canny Image",ImageCanny)

# Trackbars for Threshold 1 and 2
cv2.namedWindow("Threshold Parameters")
cv2.resizeWindow("Threshold Parameters",400,600)
cv2.createTrackbar("Threshold 1","Threshold Parameters",60,255,nothing)
cv2.createTrackbar("Threshold 2","Threshold Parameters",60,255,nothing)

# Function to get Biggest 4-sided contour 
def getBiggestcontour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area=cv2.contourArea(contour)
        if area > 2000:
            peri=cv2.arcLength(contour,True)
            shape=cv2.approxPolyDP(contour,0.02*peri,True) 
            if area > max_area and len(shape) == 4:
                biggest = shape
                max_area = area
    return biggest,max_area

def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img,(biggest[1][0][0], biggest[1][0][1]) , (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[0][0][0], biggest[0][0][1]), (0, 255, 0), thickness)
 
    return img

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

while True:
    #For Live Webcam feed, uncomment next line 
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
    #To remove the previous image windows , uncomment next line
    #cv2.destroyAllWindows()
    
    
    # Determining thresholds 1 and 2 and Applying Canny Edge Detection
    th1=cv2.getTrackbarPos("Threshold1","Threshold Parameters")  # GET TRACK BAR VALUES FOR THRESHOLDS
    th2=cv2.getTrackbarPos("Threshold2","Threshold Parameters")  
    ImageCanny=cv2.Canny(imgBlur,th1,th2)    # APPLY CANNY BLUR
    cv2.imshow("Canny Image",ImageCanny)
    cv2.waitKey(0)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(ImageCanny, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    '''
    cv2.imshow("Dilated",imgDial)
    cv2.imshow("Eroded",imgThreshold)
    cv2.waitKey()
    '''
    #To remove the previous image windows , uncomment next line
    #cv2.destroyAllWindows()
    
    # Copies the original image for Contour detection
    imgcontour1=img.copy()
    contours1, hierarchy1 =cv2.findContours(ImageCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    big,maxarea=getBiggestcontour(contours1)
    cv2.drawContours(imgcontour1,big, -1,(0, 0,255),15) 
    cv2.imshow("Contours on Original Image found on Canny Image1", imgcontour1)
    img=drawRectangle(imgcontour1,big,15)
    cv2.imshow("Contours on Original Image found on Canny Image2", img)
    cv2.waitKey(0)

    # Copies the Original image for corner Detection
    img_with_corners=img.copy()
    corners = cv2.goodFeaturesToTrack(imgBlur, 4, 0.4, 50)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_with_corners, (x, y), 4, (200, 0, 255), -1)
    cv2.imshow("Image With corners", img_with_corners)

    pts1 = np.float32(big) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[widthImg, 0],[0, 0],[0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    #REMOVE 20 PIXELS FORM EACH SIDE
    imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
    cv2.imshow("Warped Image With corners", imgWarpColored)

    im1=imgWarpColored.copy()
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im1 = cv2.filter2D(im1, -1, kernel)
    cv2.imshow("a", im1)

    im2=imgWarpColored.copy()
    im2=unsharp_mask(im2)
    cv2.imshow("b", im2)

    im3=im1.copy()
    im3=unsharp_mask(im3)
    cv2.imshow("a+b", im3)
    
    im4=imgWarpColored.copy()
    im4=cv2.GaussianBlur( im4, (5, 5), 3)
    im4=cv2.addWeighted(im4, 1.5, im4, -0.5, 0, im4)
    cv2.imshow("a+b+c", im4)

    # APPLY ADAPTIVE THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
    cv2.imshow("Warped Image With corners after adaptive theshold1", imgAdaptiveThre)

    imgWarpGray1 = cv2.cvtColor(im3,cv2.COLOR_BGR2GRAY)
    imgAdaptiveThre1= cv2.adaptiveThreshold(imgWarpGray1, 255, 1, 1, 7, 2)
    imgAdaptiveThre1 = cv2.bitwise_not(imgAdaptiveThre1)
    imgAdaptiveThre1=cv2.medianBlur(imgAdaptiveThre1,3)
    cv2.imshow("Warped Image With corners after adaptive theshold2", imgAdaptiveThre1)

    if cv2.waitKey(0) & 0xFF == ord('s'):
        break

    
    

print(cv2.__version__)
print('hi')
cv2.destroyAllWindows()