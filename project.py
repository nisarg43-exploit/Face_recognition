from tkinter import*
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
import os
from tkinter import messagebox
import cv2

import numpy as np


class face_recognition:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x765+0+0")
        self.root.title("face recognition system")
     #background image 
        img=Image.open(r"C:/Users/nisar/Documents/face_recognition/bg/background.jpg")
        img=img.resize((1530,765),Image.ANTIALIAS)
        self.photoimg=ImageTk.PhotoImage(img)
        bg_img=Label(self.root,image=self.photoimg)
        bg_img.place(x=0,y=0,width=1530,height=765)

    #photos data
        img1=Image.open(r"C:\Users\nisar\Documents\face_recognition\bg\opendata.jpg")
        img1=img1.resize((220,220),Image.ANTIALIAS)
        self.photoimg1=ImageTk.PhotoImage(img1)

        b1=Button(bg_img,image=self.photoimg1,cursor="hand2",command=self.open_img)
        b1.place(x=200,y=100,width=220,height=220)
        b1_1=Button(bg_img,text="TRANING-DATA",cursor="hand2",command=self.open_img,font=("times new roman",20,"bold"),bg="white",fg="black")
        b1_1.place(x=200,y=320,width=220,height=40)

    #prepare data
        img3=Image.open(r"C:\Users\nisar\Documents\face_recognition\bg\reshape.jpg")
        img3=img3.resize((220,220),Image.ANTIALIAS)
        self.photoimg3=ImageTk.PhotoImage(img3)
        b2=Button(bg_img,image=self.photoimg3,cursor="hand2",command=self.reshape)
        b2.place(x=450,y=100,width=220,height=220)
        b2_2=Button(bg_img,text="RESHAPE",cursor="hand2",command=self.reshape,font=("times new roman",20,"bold"),bg="White",fg="black")
        b2_2.place(x=450,y=320,width=220,height=40)

    #TRain
        img4=Image.open(r"C:\Users\nisar\Documents\face_recognition\bg\train.jpg")
        img4=img4.resize((220,220),Image.ANTIALIAS)
        self.photoimg4=ImageTk.PhotoImage(img4)
        b3=Button(bg_img,image=self.photoimg4,cursor="hand2",command=self.train_classiffier)
        b3.place(x=700,y=100,width=220,height=220)
        b3_3b3=Button(bg_img,text="TRAIN",cursor="hand2",command=self.train_classiffier,font=("times new roman",20,"bold"),bg="White",fg="black")
        b3_3b3.place(x=700,y=320,width=220,height=40)

    #detect face
        img2=Image.open(r"C:\Users\nisar\Documents\face_recognition\bg\face.jpeg")
        img2=img2.resize((220,220),Image.ANTIALIAS)
        self.photoimg2=ImageTk.PhotoImage(img2)

        b4=Button(bg_img,image=self.photoimg2,cursor="hand2",command=self.face_recog)
        b4.place(x=950,y=100,width=220,height=220)
        b4_4=Button(bg_img,text="RECOGNIZE",cursor="hand2",command=self.face_recog,font=("times new roman",20,"bold"),bg="white",fg="black")
        b4_4.place(x=950,y=320,width=220,height=40)

    #close project
        b5=Button(bg_img,text="EXIT",cursor="hand2",command=root.destroy,font=("times new roman",20,"bold"),bg="white",fg="black")
        b5.place(x=600,y=400,width=330,height=40)
#****************************************************************functions************************************************************************



#****************show data folder*************************
    def open_img(self):
        os.startfile(r"C:\Users\nisar\Documents\face_recognition\train")

#***************preparing the data********************************
    
    def reshape(self):
        
        path = ('C:/Users/nisar/Documents/face_recognition/train/')
        face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        for i, filename in enumerate(os.listdir(path)):
            os.rename("C:/Users/nisar/Documents/face_recognition/train/" + filename, "C:/Users/nisar/Documents/face_recognition/train/" + "user." + str(i) + ".jpg")

        for l, filename in enumerate(os.listdir(path)):
            img=cv2.imread('C:/Users/nisar/Documents/face_recognition/train/user.'+ str(l)+'.jpg')
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            detectface=face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in detectface:
                face_cropped=img[y:y+h,x:x+w]
                BGR2GRAY=cv2.cvtColor(face_cropped,cv2.COLOR_BGR2GRAY)
                cv2.imwrite('C:/Users/nisar/Documents/face_recognition/train/user.'+str(l)+'.jpg',BGR2GRAY)



        for item in os.listdir(path):
            if os.path.isfile(path+item):
                im = Image.open(path+item)
                f, e = os.path.splitext(path+item)
                imResize = im.resize((450,450), Image.ANTIALIAS)
                imResize.save(f + '.jpg', 'JPEG', quality=90)
        messagebox.showinfo("RESULTS","DATA PREPERATION COMPLETED!! YOU CAN TRAIN DATA")
    
#***************Train data********************************
    def train_classiffier(self):
        data_dir=(r"C:\Users\nisar\Documents\face_recognition\train")
        path=[os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces=[]
        ids=[] 

        for image in path:
            img=Image.open(image).convert('L')   ####Gray scale
            imageNP=np.array(img,'uint8')
            id=int(os.path.split(image)[1].split('.')[1])

            faces.append(imageNP)
            ids.append(id)
            cv2.imshow("Training",imageNP)
            cv2.waitKey(1)==13
        ids=np.array(ids)

#******************************Train the classifier*******************************************    
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write(r"C:\Users\nisar\Documents\face_recognition\N_classifier.xml")
        cv2.destroyAllWindows()
        messagebox.showinfo("Result","training dataset completed!!!!")

    def face_recog(self):
        
        
        def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text,clf,c):
            
            gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            features=classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
            
            list=[]
            coord=[]
            for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                id,predict=clf.predict(gray_image[y:y+h,x:x+w])
                confidence=int((100*(1-predict/300)))

                if confidence>86:
                    
                    c=c-1
                    cv2.putText(img,'MACHED!!',(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)               
                    print("No of matched person= "+str(-c))
                else:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(img,'Unknown face',(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                
                coord=[x,y,w,h]
            
            
            return coord
            
        path = ('C:/Users/nisar/Documents/face_recognition/test/')
        def recognize(img,clf,facecascade):
            c=0
            sum=0
            coord=draw_boundary(img,facecascade,1.1,10,(255,25,255),"Face",clf,c)
            
            #c=list(draw_boundary(img,facecascade,1.1,10,(255,25,255),"Face",clf,c)[:1])
            
            

            return img

        facecascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.read("C:/Users/nisar/Documents/face_recognition/N_classifier.xml")

        

        while True:
            for l, filename in enumerate(os.listdir(path)):
                img=cv2.imread('C:/Users/nisar/Documents/face_recognition/test/user.'+ str(l)+'.jpg')
                #img_main=cv2.VideoCapture(0)
                #ret, img2=img_main.read()
                
                #img=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                img=recognize(img,clf,facecascade)
                cv2.imshow("welcome to face recognition",img)
                cv2.waitKey(delay=1000)
                cv2.destroyAllWindows()
            if cv2.waitKey(delay=1000):
                break
            
            cv2.destroyAllWindows()

    

if __name__== "__main__":
    root=Tk()
    obj=face_recognition(root)
    root.mainloop()