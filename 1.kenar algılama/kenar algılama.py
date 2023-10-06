import cv2
import matplotlib.pyplot as plt 
import numpy as np


img = cv2.imread("bridge.jpg",0)
plt.figure(),plt.imshow(img, cmap= "gray"), plt.axis("off")


edges = cv2.Canny(image =img, threshold1 = 0, threshold2 = 255)
plt.figure(),plt.imshow(edges, cmap= "gray"), plt.axis("off")

mad_val = np.median(img)
print(mad_val)

low = int(max(0,(1-0.33)*mad_val))
high = int(min(255,(1+0.33)*mad_val))

print(low)
print(high)

edges = cv2.Canny(image =img, threshold1 =low,threshold2 = high)
plt.figure(),plt.imshow(edges, cmap= "gray"), plt.axis("off"), plt.show()

#blur

blurred_img = cv2.blur(img, ksize=(7,7))
plt.figure(),plt.imshow(blurred_img, cmap= "gray"), plt.axis("off")

mad_val = np.median(blurred_img)
print(mad_val)

low = int(max(0,(1-0.33)*mad_val))
high = int(min(255,(1+0.33)*mad_val))
print(low)
print(high)

edges = cv2.Canny(image =blurred_img, threshold1 =low,threshold2 = high)
plt.figure(),plt.imshow(edges, cmap= "gray"), plt.axis("off")
























