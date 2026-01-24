import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
print(os.getcwd())


# Read grayscale image
# img = cv2.imread("\Class_Wise\Histogram_Equalization\cameraman.jpg", cv2.IMREAD_GRAYSCALE)
# print(img)

# Histogram
# hist, bins = np.histogram(img.flatten(), 256, [0,256])

# # PDF
# pdf = hist / hist.sum()

# # CDF
# cdf = pdf.cumsum()

# # Normalize CDF
# cdf_normalized = np.round(cdf * 255).astype(np.uint8)

# # Apply mapping
# he_img = cdf_normalized[img]

# # Display
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
# plt.subplot(1,2,2), plt.imshow(he_img, cmap='gray'), plt.title("Histogram Equalized")
# plt.show()
