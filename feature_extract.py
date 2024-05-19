import mahotas as mt
import cv2

# Feature Extraction Methods

# Texture Using Haralick
def Texture_Extraction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    textures = mt.features.haralick(img)
    return textures.flatten()                    # Reshaping features of each image in one row

# Shape using HuMoments
def Shape_Extraction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,im1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # centre moments
    moment = cv2.moments(im1)
    # hu moments
    huMoment = cv2.HuMoments(moment)
    return huMoment.flatten()                    # Reshaping features of each image in one row

# Color using Color Histogram
def Color_Extraction(img):
    HSV_image=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([HSV_image], [0,1,2], None, [25,25,25], [0,256,0,256,0,256])
    cv2.normalize(hist,hist)
    hist=hist.flatten()                          # Reshaping features of each image in one row
    return hist