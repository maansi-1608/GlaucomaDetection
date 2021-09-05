import cv2

def gaussian_blur(image):
    blur_img = cv2.GaussianBlur(image, (41, 41), 0)
    return blur_img