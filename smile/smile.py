import numpy as np
import pandas
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2


#This code that i wrote blow take photo, Grayscale it and mirror it
#cam_port = 0
#cam = cv2.VideoCapture(cam_port)
#takenpic, image = cam.read()
#cv2.imshow("smile", image)   #show original image
#cv2.imwrite("Smile.png", image)    #save pic
#mirror = cv2.flip(image, 1)
#cv2.imshow("reverese", mirror) #reverese
#cv2.imwrite("reverese.png", mirror)   #save reverse
#gray original and mirror
#graypicm = cv2.cvtColor(mirror, cv2.COLOR_BGR2GRAY)
#graypic = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", graypic)
#cv2.imwrite("gray.png", graypic)
#cv2.imshow("gray1.png", graypicm)
#cv2.imwrite("gray1.png", graypicm)
#cv2.waitKey(0)


#this code take a photo when you smile infront of webcam
video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
while True:
    success, img = video.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)
    cnt = 500
    keyPressed = cv2.waitKey(1)
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
        smile = smile_cascade.detectMultiScale(grayImg, 1.8, 15)
        for x, y, w, h in smile:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 5)
            print("Image " + str(cnt) + "Saved")
            path = r'/home/marjan/Desktop/Projects/first-env' + str(cnt) + '.jpg'
            cv2.imwrite(path, img)
            cnt = cnt+1
            if (cnt >= 503):
                break

    cv2.imshow('live video', img)
    if (keyPressed & 0xFF == ord('q')):
        break

video.release()
#cv2.destroyAllWindows()


