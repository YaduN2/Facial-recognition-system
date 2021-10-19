import cv2
import numpy as np
import face_recognition
import os

path = 'IMAGES'
MyList = os.listdir(path)
images = []
classImg = []
for img in MyList:
    currentImg = cv2.imread(f'{path}/{img}')
    images.append(currentImg)
    classImg.append(os.path.splitext(img)[0])
print(classImg)

def findencoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistKNOWN = findencoding(images)
print('ENCODING DONE!')

encodelistKNOWN = findencoding(images)
print(len(encodelistKNOWN))

video = cv2.VideoCapture(0)
while True:
    success, img = video.read()

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    FacesFound = face_recognition.face_locations(imgSmall)
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall,FacesFound)


    for enocodeFace,facloc in zip(encodeCurrentFrame,FacesFound):
        mathes = face_recognition.compare_faces(encodelistKNOWN,enocodeFace)
        facedistance = face_recognition.face_distance(encodelistKNOWN,enocodeFace)
        print(facedistance)
        matchIndex = np.argmin(facedistance)
        print(matchIndex)
        if mathes[matchIndex]:
            name = classImg[matchIndex]
            print(name)
            y1, x2, y2, x1 = facloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', img)
    cv2.waitKey(1)

# imgCase = face_recognition.load_image_file("IMAGES/test_3.jpeg")
# imgCase = cv2.cvtColor(imgCase, cv2.COLOR_BGR2RGB)
# faceLoc = face_recognition.face_locations(imgCase)[0]
# encodeface = face_recognition.face_encodings(imgCase)[0]
# imgTest = face_recognition.load_image_file("IMAGES/test_4.jpeg")
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodefaceTest = face_recognition.face_encodings(imgTest)[0]
#
# cv2.rectangle(imgCase, (faceLoc[3], faceLoc[2]), (faceLoc[1], faceLoc[0]), (255, 0, 0), 2)
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[2]), (faceLocTest[1], faceLocTest[0]), (255, 0, 0), 2)
#
# compare = face_recognition.compare_faces([encodeface], encodefaceTest)
# facedistance = face_recognition.face_distance([encodeface], encodefaceTest)
# print(compare)
# print(facedistance)
#
#
# cv2.imshow("CASE", imgCase)
# cv2.imshow("TEST", imgTest)
#
# cv2.waitKey(0)
