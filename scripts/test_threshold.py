from cps_maze.config import load_config

import cv2
import matplotlib.pyplot as plt
import numpy as np

config = load_config("configs/default.yaml")
vision_cfg = config.vision

image = cv2.imread("calibration/threshold_test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(
    gray,
    (int(vision_cfg["blur_kernel"]) | 1, int(vision_cfg["blur_kernel"]) | 1),
    0,
)

def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_multiple(img, title, ax):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

# show_image(image, "Original Grayscale Image")

thresh_mean = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5
)
# show_image(thresh_mean, "Thresh Mean")
#blurred
# 11, 10 -> dookie
# 21, 10 -> no details, can't make out ball
# 11, 2 -> grainy walls, kinda good ball
# 21, 2 -> Thick lines, dark ball
#no blur
# 21, 2 -> dark, lot of grain, ball kinda clear
# 11, 2 -> heavy grain, hard to make out

thresh_gauss = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
)
# show_image(thresh_mean, "Thresh Gaussian")

_, binary = cv2.threshold(
    blurred,
    int(vision_cfg["threshold_value"]),
    255,
    cv2.THRESH_BINARY,
)

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=100,
    param2=25,
    minRadius=5,
    maxRadius=30,
)

circles_test = cv2.cvtColor(thresh_mean, cv2.COLOR_GRAY2BGR)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(circles_test, (x, y), r, (0, 255, 0), 2)
        cv2.circle(circles_test, (x, y), 2, (0, 0, 255), 3)

show_image(circles_test, "Hough Circles")

# fig, axes = plt.subplots(1, 5, figsize=(20, 5))
# for ax, img, title in [
#     (axes[0], gray, "Gray"),
#     (axes[1], blurred, "Blurred"),
#     (axes[2], binary, "Binary Threshold"),
#     (axes[3], thresh_mean, "Adaptive Mean"),
#     (axes[4], thresh_gauss, "Adaptive Gaussian"),
# ]:
#     show_multiple(img, title, ax)

# plt.tight_layout()
# plt.show()