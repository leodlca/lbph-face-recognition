import cv2
import os
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create(threshold=500)

def get_img_names():
    paths = [os.path.join('faces', p) for p in os.listdir('faces')]

    faces = []
    ids = []

    for path in paths:
        face_img = cv2.imread(path)
        face_img_grey = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        dir, id, mili, ext = path.split('.')

        faces.append(face_img_grey)
        ids.append(int(id))

    return np.array(ids), faces

ids, faces = get_img_names ()

print('Training has started!')

lbph.train(faces, ids)
lbph.write('classifiers/lbphClassifier.yml')

print('Finished training!')