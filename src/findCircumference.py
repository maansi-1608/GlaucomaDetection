import cv2

def findCircumference(image):
    contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] #if imutils.is_cv2() else contours[1]
    diameter = max(contours, key=cv2.contourArea)
    return diameter