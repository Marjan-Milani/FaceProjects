from face_recognition import face_encodings, compare_faces
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

def recognise (img1, img2):
    img_encoding1 = readImage(img1)
    img_encoding2 = readImage(img2)
    return compare_faces([img_encoding1], img_encoding2)

def readImage(image):
    img = cv2.imread(image)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return face_encodings(rgb_img)[0]


# if use sample image you see not ok because there isnt any similar photo on
#  the img folder, but if you use 1.jpg image you can see the first similar picture
# in a figure window

ref_img = 'sample.JPG'
ref_dir_path = '/home/marjan/Desktop/Projects/first-env/face/CompareFace/img/'
lst_img = glob.glob(ref_dir_path + '*')
print(lst_img)
temp = False
for i in lst_img:
    if(recognise(i, ref_img)[0]):
        temp = True
        break
img1 = cv2.imread(ref_img)
img1 = cv2.resize(img1, (320, 240))   
if(temp):
    img2 = cv2.imread(i)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0])) 
    numpy_vertical = np.hstack((img1, img2))
    cv2.imshow("OK", numpy_vertical)
else:
    img2 = cv2.imread('img2.png')
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0])) 
    numpy_vertical = np.hstack((img1, img2))
    cv2.imshow("Not OK", numpy_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()
