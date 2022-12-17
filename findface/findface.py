import numpy as np
import pandas
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2
image = cv2.imread("path")
image1 = cv2.resize(image, (400, 400))
grayImg = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("B1", grayImg)
cv2.waitKey(0)
cv2.imwrite("path", grayImg)

#take pic from webcam
cam_port = 0
cam = cv2.VideoCapture(0)
takenpic, image = cam.read()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image = cv2.imread("path")
#resize the pic
image1 = cv2.resize(image, (400, 400))
#print the shape of pic
imageToMatrice = np.asarray(image1)
print(imageToMatrice.shape)
#detect the face
while True:
    grayImg = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)
    cnt = 500
    keyPressed = cv2.waitKey(0)
    for x, y, w, h in faces:
        image1 = cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 200), 3)
        print("Image " + str(cnt) + "Saved")
        path = r'/save path' + str(cnt) + '.jpg'
        cv2.imwrite(path, image)
        cnt = cnt+1
        if (cnt >= 503):
            break
    cv2.imshow('img', image1)
cam.release()
cv2.destroyAllWindows()