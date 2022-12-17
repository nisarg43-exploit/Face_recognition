from tkinter import*
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
from tkinter import messagebox
import cv2
import os
import numpy as np

def face_recog(self):
        
    def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text,clf,c=0):
            img=cv2.VideoCapture(0)
            gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            features=classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
            
            list=[]
            coord=[]
            for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                id,predict=clf.predict(gray_image[y:y+h,x:x+w])
                confidence=int((100*(1-predict/300)))

                if confidence>85:
                    cv2.putText(img,'namrata',(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    print(c)
                    cv2.imshow("welcome to face recognition",img)
                    cv2.waitKey()
        
                else:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(img,'Unknown face',(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)

                coord=[x,y,w,h]
            
            
            return coord
    
    def recognize(img,clf,facecascade):
            coord=draw_boundary(img,facecascade,1.1,10,(255,25,255),"Face",clf)
            return img

    facecascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.read("C:/Users/nisar/Documents/face_recognition/N_classifier.xml")

        

    
            
            
            
            
            
           
                

























