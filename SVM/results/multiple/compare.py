
"""Compare two aligned images of the same size.
Usage: python compare.py first-image second-image
"""

import sys

from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

import cv2

from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error
import matplotlib.image as mpimg

def main():
    file1, file2 = sys.argv[1:1+2]
    # read images as 2D arrays (convert to grayscale for simplicity)
    img1 = to_grayscale(imread(file1).astype(float))
    img2 = to_grayscale(imread(file2).astype(float))
    # compare
    n_m, n_e, n_0 = compare_images(img1, img2)
    ssim_val = find_ssim(file1, file2)
    rmse_val = find_rmse(file1, file2)
    print ("L0 norm per pixel:", n_0*1.0/img1.size)
    print ("L1 norm per pixel:", n_m/img1.size)
    print ("L2 norm per pixel:", n_e/img1.size)
    print ("SSIM:", ssim_val)
    print ("RMSE:", rmse_val)


def compare_images(img1, img2):
    # normalize to compensate for exposure difference
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    e_norm = sum((diff) ** 2)  # Euclidean norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    
    return (m_norm, e_norm, z_norm)


def find_ssim(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    return ssim(img1, img2, multichannel=True)

def find_rmse(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1_r = img1[:,:,2]
    img1_g = img1[:,:,1]
    img1_b = img1[:,:,0]
    img2_r = img2[:,:,2]
    img2_g = img2[:,:,1]
    img2_b = img2[:,:,0]
    
    rmse = mean_squared_error(img1_r, img2_r) ** 0.5
    rmse += (mean_squared_error(img1_g, img2_g) ** 0.5)
    rmse += (mean_squared_error(img1_b, img2_b) ** 0.5)
    rmse /= 3

    return rmse

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

if __name__ == "__main__":
    main()

