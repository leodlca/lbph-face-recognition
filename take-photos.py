import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# UI

id_names = pd.read_csv('id-names.csv')

print('Welcome!')
print('\nIf this is your first time, choose a random ID number from 1 to 100000.')
print('If it\'s not, please type your ID.')

id = int(input('ID: '))

if id_names[id_names['id'] == id].size == 0:
    name = input('Now please tell me your name: ')
    id_names = id_names.append({'id': id, 'name': name}, ignore_index=True)
    id_names.to_csv('id-names.csv')
    print('\nThanks {0}, let\'s do this!'.format(name))
else:
    name = id_names[id_names['id'] == id]['name'][0]
    print('\nWelcome back, {0}, let\'s do this!'.format(name))

camera_id = input('\nPlease input your camera ID (the default is 0, just press enter if you don\'t know): ')

print('This is a very important step, what you have to do is press the \'s\' key repeatedly when you see a red \
rectangle around your face. I recommend you to take from 25 up until 100 photos from different angles, illumination, \
with and without glasses, smiling, sleepy, looking down, eyes closed, you get it.')
input('\nPress ENTER to start when you\'re ready, and press the \'q\' key to quit when you\'re done!')

# Face Detection and Photo Capturing

face_classifier = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt.xml')

camera = cv2.VideoCapture(0 if len(camera_id) == 0 else int(camera_id))

LARGURA, ALTURA, photos_taken = 220, 220, 0

if not os.path.exists('faces'):
    os.makedirs('faces')

while cv2.waitKey(1) != ord('q'):
    connected, img = camera.read()
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_classifier.detectMultiScale(grey_img, scaleFactor=1.1, minSize=(50,50), minNeighbors=5)

    for x, y, w, h in detected_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        face_region = img[y:y + h, x:x + w]
        face_region_grey = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) & 0xFF == ord('s') and np.average(grey_img) > 50:
            face_img = cv2.resize(grey_img[y:y + h, x:x + w], (LARGURA, ALTURA))
            img_name = 'face.{0}.{1}.jpg'.format(id, str(datetime.now().microsecond))
            cv2.imwrite('faces/{0}'.format(img_name), face_img)

            photos_taken += 1
            print('-> {0} photo(s) taken!'.format(str(photos_taken)))

    cv2.imshow("Face", img)

camera.release()
cv2.destroyAllWindows()