import cv2 # type: ignore
import numpy as np #type: ignore

# Load image
img = cv2.imread('/home/snehal/GIT-HUB/License-PLate-Detection-Model/train/images/TEST.jpeg')
h, w = img.shape[:2]

# Your annotation
center_x, center_y, width, height = 0.499, 0.701, 0.301, 0.101

# Convert to pixel coordinates
x_center = int(center_x * w)
y_center = int(center_y * h)
box_width = int(width * w)
box_height = int(height * h)

# Calculate corners
x1 = int(x_center - box_width/2)
y1 = int(y_center - box_height/2)
x2 = int(x_center + box_width/2)
y2 = int(y_center + box_height/2)

# Draw bounding box
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('Annotation Check', img)
cv2.waitKey(0)