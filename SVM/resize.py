import cv2

img = cv2.imread('./results/multiple/train2.jpg')
img = cv2.resize(img,(150,100))
#img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
cv2.imwrite('./results/multiple/train2.jpg',img)
