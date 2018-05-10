'''

Run as python3 confusion_matrix.py gold_file test_file output_file
All dimensions should be 256x256
If not, scale your images or modify the for loops to iterate over the image

'''


import cv2
from skimage.measure import compare_ssim as ssim
import matplotlib.image as mpimg
import os
import sys



color = {0:'red', 1:'green', 2:'blue'}

def compare_images(imageA, imageB):
    s = ssim(imageA, imageB)
#     print("compare_images: %.2f" % (s))
    return s

def get_confusion_matrix(C, gold, test, result):
    
    goldR = gold[:,:,2]
    resultR = result[:,:,2]
    
    goldG = gold[:,:,1]
    resultG = result[:,:,1]
    
    goldB = gold[:,:,0]
    resultB = result[:,:,0]
    
    sR_thresh = compare_images(goldR, test)
    sR_value = compare_images(goldR, resultR)
    
    sG_thresh = compare_images(goldG, test)
    sG_value = compare_images(goldG, resultG)
    
    sB_thresh = compare_images(goldB, test)
    sB_value = compare_images(goldB, resultB)
    
    
    s = [sR_thresh, sG_thresh, sB_thresh]
    actual_index = s.index(max(s))
    
#     print (s[actual_index])
    
    s_test = [sR_value, sG_value, sB_value]
    pred_index = s_test.index(max(s_test))
#     print (s)
#     print (s_test)
    
#     print (s_test[pred_index])
    
#     print (color[actual_index], color[pred_index])
    
    C[actual_index][pred_index] += 1 
    
    
    return C

C = np.zeros((3,3))
gold = cv2.imread(sys.argv[1])
image = mpimg.imread(sys.argv[1])

test = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2GRAY)
result = cv2.imread(sys.argv[3])

y = 0
x = 0

for i in range(32):
    for j in range(32):
        crop_gold = gold[y:y+7, x:x+7]
        crop_test = test[y:y+7, x:x+7]
        crop_result = result[y:y+7, x:x+7]
        C = get_confusion_matrix(C, crop_gold, crop_test, crop_result)
        x += 8
    y += 8
    x = 0
    
print (C)
