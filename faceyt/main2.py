import cv2
import face_recognition

imgelon=face_recognition.load_image_file('images/elon1.jpg')
imgelon=cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
testelon=face_recognition.load_image_file('elontest.jpg')
testelon=cv2.cvtColor(testelon,cv2.COLOR_BGR2RGB)



facelocation=face_recognition.face_locations(imgelon)[0]#gives 4 values for rectangle
encodeelon=face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(facelocation[3],facelocation[0]),(facelocation[1],facelocation[2]),(255,0,255),2)

facelocationtest=face_recognition.face_locations(testelon)[0]#gives 4 values for rectangle
encodeelontest=face_recognition.face_encodings(testelon)[0]
cv2.rectangle(testelon,(facelocationtest[3],facelocationtest[0]),(facelocationtest[1],facelocationtest[2]),(255,0,255),2)

#we got the128x128 encoding of both the  images now we compate it with linear svm

facedistance=face_recognition.face_distance([encodeelon],encodeelontest)#lower the distance better the match is
print(facedistance)


results=face_recognition.compare_faces([encodeelon],encodeelontest)
cv2.putText(testelon,f'{results}{round(facedistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

print(results)


cv2.imshow('elonmusk',imgelon)
cv2.waitKey(0)
cv2.imshow('elonmusk',testelon)
cv2.waitKey(0)