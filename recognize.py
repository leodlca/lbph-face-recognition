import cv2
import pandas as pd

id_names = pd.read_csv('id-names.csv')

WIDTH, HEIGHT, FONT, SUB_FONT = 220, 220, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL

face_detector = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=500)
face_recognizer.read('classifiers/lbphClassifier.yml')

camera = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord('q'):
    connected, img = camera.read()

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(img_grey, scaleFactor=1.1, minSize=(50, 50))

    for x, y, w, h in detected_faces:
        img_face = cv2.resize(img_grey[y:y + h, x:x + w], (WIDTH, HEIGHT))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

        id, trust = face_recognizer.predict(img_face)
        if id != -1:
            try:
                cv2.putText(img, id_names[id_names['id'] == id].iloc[0]['name'], (x, y + h + 30), FONT, 1, (0, 0, 255))
                cv2.putText(img, str(trust), (x, y + h + 60), FONT, 0.5, (0, 0, 255))
            except:
                pass

    cv2.imshow("Recognize", img)

camera.release()
cv2.destroyAllWindows()