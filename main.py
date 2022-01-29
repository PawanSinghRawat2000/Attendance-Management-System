import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='images'
images=[]
personName=[]
mylist=os.listdir(path)
#print(mylist)

#image reading and extracting persons name
for cu_image in mylist:
    current_image=cv2.imread(f'{path}/{cu_image}')
    images.append(current_image)
    personName.append(os.path.splitext(cu_image)[0])
#print(personName)


#encoding images using facerecognition
def faceEncodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
encodelistKnown=faceEncodings(images)
print("All encodings complete")

def attendance(name):
    with open('attendance.csv','r+') as f:
        myDatalist=f.readlines()
        nameList=[]
        for line in myDatalist:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now=datetime.now()
            timestr=time_now.strftime('%H:%M:%S')
            datestr=time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{timestr},{datestr}')


cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    faces = cv2.resize(frame,(0,0),None,0.25,0.25)
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)

    faces_currentFrame=face_recognition.face_locations(faces)
    encodesCurrentFrame=face_recognition.face_encodings(faces,faces_currentFrame)

    for encodeface,faceloc in zip(encodesCurrentFrame,faces_currentFrame):
        matches=face_recognition.compare_faces(encodelistKnown,encodeface)
        faceDist=face_recognition.face_distance(encodelistKnown,encodeface)

        matchIndex=np.argmin(faceDist)

        if(matches[matchIndex]):
            name=personName[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceloc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            attendance(name)
    cv2.imshow("Camera",frame)
    if cv2.waitKey(10)==13:
        break
cap.release()
cv2.destroyAllWindows()



