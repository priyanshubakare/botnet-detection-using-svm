

import cv2  
import numpy as np
  

# We load it with imread command  
picture = cv2.imread('C:/gear_dataset/Defected gear/1.jpg')  



import cv2  
import numpy as np
 
img = cv2.imread('C:/gear_dataset/Non defected gear/1.jpg') 
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
 
ret, thresh_hold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) 
ret, thresh_hold1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV) 
ret, thresh_hold2 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO) 
ret, thresh_hold3 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV) 
ret, thresh_hold4 = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)   
 
thresh_hold = cv2.resize(thresh_hold, (960, 540))    
cv2.imshow('Binary Threshold Image', thresh_hold) 
 
thresh_hold1 = cv2.resize(thresh_hold1, (960, 540))    
cv2.imshow('Binary Threshold Inverted Image', thresh_hold1) 
 
thresh_hold2 = cv2.resize(thresh_hold2, (960, 540))    
cv2.imshow('Threshold Tozero Image', thresh_hold2) 
 
thresh_hold3 = cv2.resize(thresh_hold3, (960, 540))    
cv2.imshow('ThresholdTozero Inverted output', thresh_hold3) 
 
thresh_hold4= cv2.resize(thresh_hold4, (960, 540))    
cv2.imshow('Truncated Threshold output', thresh_hold4) 
 
if cv2.waitKey(0) & 0xff == 25:  
    cv2.destroyAllWindows()