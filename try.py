import numpy as np
import matplotlib.pyplot as plt
import cv2


def create_gaussian(window_size:int, sigma:float = 1.0):
    x, y = np.meshgrid(
        np.linspace(-(window_size//2), window_size//2, window_size),
        np.linspace(-(window_size//2), window_size//2, window_size)
    )
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian / np.sum(gaussian)

def harris(gray:np.ndarray):
    gaussian = create_gaussian(5)
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy
    ixx = cv2.filter2D(Ix2, -1, gaussian)
    iyy = cv2.filter2D(Iy2, -1, gaussian)
    ixy = cv2.filter2D(Ixy, -1, gaussian)
    det = ixx * iyy - ixy**2
    trace = ixx + iyy
    response = det - 0.04 * trace**2
    return response

image = plt.imread('chessboard.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
original_image = image.copy()

nilay_response = harris(gray)
rows, cols = gray.shape
for r in range(rows):
    for c in range(cols):
        if nilay_response[r, c] > 0.1:
            cv2.circle(gray, (c,r), 1, (255, 0, 0), -1)


harris_cv = original_image.copy()
harris_response = cv2.cornerHarris(gray, 2, 3, 0.04)
for r in range(rows):
    for c in range(cols):
        if harris_response[r, c] > 0.01:
            cv2.circle(harris_cv, (c,r), 1, (0, 255, 0), 1)

# plt.subplot(2,2,1)
# plt.title("Original Image")
# plt.imshow(original_image)
# plt.subplot(2,2,2)
# plt.title("Gray Image")
# plt.imshow(gray, cmap='gray')
# plt.subplot(2,2,3)
# plt.title("My Harris Response")
# plt.imshow(image)
# plt.subplot(2,2,4)
# plt.title("OG Harris Response")
# plt.imshow(harris_cv)
# plt.show()
plt.imshow(gray,cmap="gray")
plt.title("My Harris Response")
plt.show()