# main.py
import numpy as np
from PIL import ImageGrab
import time
from cv2 import cv2
import pytesseract
import utilities

# Define all the global variables
heightImg=640
widthImg=480
Image_Resolution_height=640
Image_Resolution_width=480

# For Photo from live Webcam feed
#img_counter=utilities.webcamfeed(heightImg,widthImg,Image_Resolution_height,Image_Resolution_width)
img_counter=2
#For Live Webcam feed, uncomment next line 
    #if webCamFeed:success, img = cap.read()
    #else:

# Reading the Image
# Filepath of Image 
pathImage = "opencv_frame_{}.png".format(img_counter-1) 
img=cv2.imread(pathImage)
heightImg, widthImg,_ = img.shape

# Trackbars for Threshold 1 and 2
cv2.namedWindow("Threshold Parameters")
cv2.resizeWindow("Threshold Parameters",400,600)
cv2.createTrackbar("Threshold 1","Threshold Parameters",60,255,utilities.nothing)
cv2.createTrackbar("Threshold 2","Threshold Parameters",60,255,utilities.nothing)

Warped_Img=utilities.process(img,heightImg, widthImg)
utilities.adaptive_thresholding(Warped_Img,heightImg,widthImg)