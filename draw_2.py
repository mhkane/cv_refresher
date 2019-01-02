import numpy as np
import cv2
from matplotlib import pyplot as plt

# Create a black image
img = np.zeros((512,512,3), np.uint8)


#Draw ellipse1 green
cv2.ellipse(img,(256,256),(25,25),0,0,300,(0,255,0),-1)

#Draw ellipse2 blue
cv2.ellipse(img,(356,256),(25,25),-60,0,300,(0,0,255),-1)


#Draw ellipse3 red
cv2.ellipse(img,(306,169),(25,25),120,0,300,255,-1)


#Draw a circle inside of rectangle
cv2.circle(img,(256,256), 10, (0,0,0), -1)

#Draw a circle inside of rectangle
cv2.circle(img,(356,256), 10, (0,0,0), -1)

#Draw a circle inside of rectangle
cv2.circle(img,(306,169), 10, (0,0,0), -1)


plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()