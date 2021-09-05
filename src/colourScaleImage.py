import displayImage 
import cv2 

def red_scale_image(img1):
    img1[:,:,1]=0
    img1[:,:,0]=0
    displayImage.display_image(img1)
    return(img1)

def green_scale_image(img2):
    img2[:,:,2]=0
    img2[:,:,0]=0
    displayImage.display_image(img2)
    return(img2)

def greyscale(image):
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_img