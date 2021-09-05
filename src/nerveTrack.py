import cv2 
import centerCrop as crop
import colourScaleImage as colour
import math 
import displayImage

def nerve_track(th2):
    l = th2.shape[0]
    w = th2.shape[1]
    i = 0
    j = 0
    x=[]
    nerve = []
    for i in range(4):
        th2 = cv2.rotate(th2, cv2.cv2.ROTATE_90_CLOCKWISE) 
        while(i<l/2):
            j = 0
            while(j<w/2):
                if th2[i][j] == 255:
                    if [i,j] in nerve:
                        x.append([i,j])
                        break
                    i += 1 
                    nerve.append([i,j])
                j+=1
            i+=1
    #print(x)  
    return x

def nerve(image):
    dim = [200,200]
    #dim.append(image.shape[0] - 75)
    #dim.append(image.shape[1] - 75)
    cropped_img = crop.center_crop(image,dim)
    gray = colour.greyscale(cropped_img)
    displayImage.display_image(gray)
    gray = cv2.medianBlur(gray,5)
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
    displayImage.display_image(th2)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(th2)
    print(maxVal)
    common_points = nerve_track(th2)
    width, height = th2.shape[1], th2.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    i=0
    len_arr = []
    for i in range(0,len(common_points)):
        length = math.sqrt((common_points[i][0]-mid_x)**2+(common_points[i][1]-mid_y)**2)
        len_arr.append(length)
        i=i+1
    if(len(len_arr)!=0):
        min_value = min(len_arr)
        min_index = len_arr.index(min_value)
        nerve_center = common_points[min_index]
        return nerve_center