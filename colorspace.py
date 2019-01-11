import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    threshold = 15
    #threshold the image to only get the blue color
    blue = np.uint8([[[255,0,0 ]]])
    hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
    lower_blue = hsv_blue.copy()
    lower_blue[0][0][0]-=threshold
    higher_blue = hsv_blue.copy()
    higher_blue[0][0][0]+=threshold
    blue_mask = cv2.inRange(hsv, lower_blue, higher_blue)

    #threshold the image to only get the green color
    green = np.uint8([[[0,255,0 ]]])
    hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    lower_green = hsv_green.copy()
    lower_green[0][0][0]-=threshold
    higher_green = hsv_green.copy()
    higher_green[0][0][0]+=threshold
    green_mask = cv2.inRange(hsv, lower_green, higher_green)

    #threshold the image to only get the red color
    red = np.uint8([[[0,0,255 ]]])
    hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
    lower_red = hsv_red.copy()
    lower_red[0][0][0]-=threshold
    higher_red = hsv_red.copy()
    higher_red[0][0][0]+=threshold
    blue_mask = cv2.inRange(hsv, lower_red, higher_red)


    # Threshold the HSV image to get only blue colors

    both_mask = cv2.bitwise_or(green_mask,blue_mask)
    three_mask = cv2.bitwise_or(both_mask,red_mask)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= three_mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()



