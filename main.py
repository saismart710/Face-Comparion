import cv2
import numpy as np
import face_recognition


#step 1 Loading and viewing image
imgme = face_recognition.load_image_file('Res/me.jpg')
imgme = cv2.cvtColor(imgme,cv2.COLOR_BGR2RGB)
imgmetest = face_recognition.load_image_file('Res/jely.jpg')
imgmetest = cv2.cvtColor(imgmetest,cv2.COLOR_BGR2RGB)

#step 2 facelocating & squaring
faceLoc = face_recognition.face_locations(imgme)[0]
encodeme = face_recognition.face_encodings(imgme)[0]
cv2.rectangle(imgme,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLoc = face_recognition.face_locations(imgmetest)[0]
encodemetest = face_recognition.face_encodings(imgmetest)[0]
cv2.rectangle(imgmetest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeme],encodemetest)
faceDis = face_recognition.face_distance([encodeme],encodemetest)
print(result,faceDis)
cv2.putText(imgmetest,f'{result}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,)



cv2.imshow('Me',imgme)
cv2.imshow('MeTest',imgmetest)
cv2.waitKey(0)