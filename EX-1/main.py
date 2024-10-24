
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
img = cv2.imread('input_image.jpg')

negative_img = 255 - img

def increase_contrast(img):
    alpha = 2.0  # Độ sáng
    beta = 0     # Độ tương phản
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

contrast_img = increase_contrast(img)

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image

log_img = log_transform(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
equalized_img = cv2.equalizeHist(img_gray)

plt.subplot(221), plt.imshow(cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)), plt.title('Negative Image')
plt.subplot(222), plt.imshow(cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB)), plt.title('Increased Contrast')
plt.subplot(223), plt.imshow(log_img, cmap='gray'), plt.title('Log Transform')
plt.subplot(224), plt.imshow(equalized_img, cmap='gray'), plt.title('Histogram Equalization')
plt.show()

cv2.imwrite('negative_image.jpg', negative_img)
cv2.imwrite('contrast_image.jpg', contrast_img)
cv2.imwrite('log_image.jpg', log_img)
cv2.imwrite('equalized_image.jpg', equalized_img)
