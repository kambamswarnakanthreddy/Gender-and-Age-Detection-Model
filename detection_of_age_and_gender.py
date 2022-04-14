from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

pathf = 'C:\\Users\\kamba\\Anaconda\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
face_classifier=cv2.CascadeClassifier(pathf)
age_model = load_model('age_model_50epochs.h5')
gender_model = load_model('gender_model_50epochs.h5')

gender_labels = ['Male', 'Female']

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    labels=[]
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        roi=roi_gray.astype('float')/255.0  #Scale
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)  

        #Gender
        roi_color=frame[y:y+h,x:x+w]
        roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
        gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
        gender_label=gender_labels[gender_predict[0]] 
        gender_label_position=(x,y-40) 
        cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        #Age
        age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        age = round(age_predict[0,0])
        age_label_position=(x,y-10)
        cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
    cv2.imshow('Gender and Age Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
