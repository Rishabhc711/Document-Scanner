import cv2
import pytesseract

# Image to String Code Snippet
heightImg = 640
widthImg  = 480 
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('Images/20210114_230948.jpg')
img = cv2.resize(img, (widthImg, heightImg))
print(img)
cv2.imshow("input",img)
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("after colour conversion", img)
print(pytesseract.image_to_string(img))
cv2.waitKey(0)
cv2.destroyAllWindows()