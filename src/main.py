import imutils
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import kMeans as km
import colourScaleImage as colour
import displayImage
import nerveTrack as nt
import findCircumference as circum
import gaussianBlur as gb
import centerCrop as crop
import elbowmethod as em

image = cv2.imread(r"./assets/(32).jpg")
grey_img = colour.greyscale(image)
displayImage.display_image(grey_img)

displayImage.display_image(crop.center_crop(image,[200,200]))

image1=image.copy()
image2=image.copy()

nerve_center = nt.nerve(image2)
print(nerve_center)


(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gb.gaussian_blur(grey_img))
img1 = image1[maxLoc[1]-100:maxLoc[1]+100, maxLoc[0]-100:maxLoc[0]+100]
img2 = img1.copy()


# Getting red image
red_image = colour.red_scale_image(img1)
# Getting green image
green_image = colour.green_scale_image(img2)

img1 = colour.greyscale(red_image)
img2 = colour.greyscale(green_image)

disc = km.kMeans(img1,3)
disc = cv2.morphologyEx(disc, cv2.MORPH_OPEN, (5,5), iterations=0)
displayImage.display_image(disc)
cup = km.kMeans(img2,4)
cup = cv2.morphologyEx(cup, cv2.MORPH_CLOSE, (5,5), iterations=0)
displayImage.display_image(cup)


contours,hierarchy = cv2.findContours(disc,2,1)
cnt = contours[0]


if nerve_center is not None:
    if cv2.pointPolygonTest(cnt,(nerve_center[0],nerve_center[1]),False) == -1 and (disc[nerve_center[0]][nerve_center[1]] == 0) and math.sqrt((nerve_center[0]-maxLoc[0])**2+(nerve_center[1]-maxLoc[1])**2)>5 :
        print("True")
    else:  
        image_correction = crop.center_crop(image2,[200,200])
        image_correction2 = image_correction.copy()
        
        red_correction = colour.red_scale_image(image_correction)
        green_correction = colour.green_scale_image(image_correction2)
        img1 = colour.greyscale(red_correction)
        img2 = colour.greyscale(green_correction)
        disc = km.kMeans(img1,3)
        disc = cv2.morphologyEx(disc, cv2.MORPH_OPEN, (5,5), iterations=0)
        displayImage.display_image(disc)
        cup = km.kMeans(img2,4)
        cup = cv2.morphologyEx(cup, cv2.MORPH_CLOSE, (5,5), iterations=0)
        displayImage.display_image(cup)

disc_diameter = -1
cup_diameter = -1
for i in range(0,360,1):
    circumference1 = circum.findCircumference(disc)
    extLeft = tuple(circumference1[circumference1[:, :, 0].argmin()][0])
    extRight = tuple(circumference1[circumference1[:, :, 0].argmax()][0])
    if(disc_diameter<abs(extRight[0]-extLeft[0])):
        disc_diameter = abs(extRight[0]-extLeft[0])
    rotated = imutils.rotate_bound(disc, angle=i)
    circumference2 = circum.findCircumference(cup)
    extLeft = tuple(circumference2[circumference2[:, :, 0].argmin()][0])
    extRight = tuple(circumference2[circumference2[:, :, 0].argmax()][0])
    if(cup_diameter<abs(extRight[0]-extLeft[0])):
        cup_diameter = abs(extRight[0]-extLeft[0])
    rotated = imutils.rotate_bound(cup, angle=i)
cup_radius = cup_diameter/2
disc_radius = disc_diameter/2
area_cup = 2*np.pi*(cup_radius**2)
area_disc = 2*np.pi*(cup_diameter**2)
ratio = cup_diameter/disc_diameter

print("cup_diameter" , cup_diameter)
print("disc_diameter",disc_diameter)
print("diameter difference" , disc_diameter - cup_diameter)
print("area_disc" , area_disc)
print("area_cup", area_cup)
print("area difference" , area_disc - area_cup)
print("Ratio",ratio)
em.elbow_method(colour.greyscale(red_image))
em.elbow_method(colour.greyscale(green_image))
displayImage.display_image(colour.greyscale(red_image))
displayImage.display_image(colour.greyscale(green_image))
plt.hist(colour.greyscale(red_image).ravel(),256,[0,256]);
plt.xlabel("Pixel values")
plt.ylabel("Number of pixels")
plt.show()
plt.hist(colour.greyscale(green_image).ravel(),256,[0,256]);
plt.xlabel("Pixel values")
plt.ylabel("Number of pixels")
plt.show()
