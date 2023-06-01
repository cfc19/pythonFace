import os
import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()

face_ids = -1
person_names = ""
x_train = []
y_ids = []

face_images = os.path.join(os.getcwd(), "Face_Images")
print(face_images)

for root, dirs, files in os.walk(face_images):
    for file in files:
        if file.endswith((".jpeg", ".jpg", ".png")):
            path = os.path.join(root, file)
            person_name = os.path.basename(root)
            print(path, person_name)

            if person_name != person_names[-1]:
                face_ids += 1
                person_names.append(person_name)

            gray_image = Image.open(path).convert("L")
            crop_image = gray_image.resize((800, 800), Image.ANTIALIAS)
            final_image = np.array(crop_image, "uint8")
            faces = face_cascade.detectMultiScale(final_image, scaleFactor=1.5, minNeighbors=5)
            print(face_ids, faces)

            for (x, y, w, h) in faces:
                roi = final_image[y:y+h, x:x+w]
                x_train.append(roi)
                y_ids.append(face_ids)

recognizer.train(x_train, np.array(y_ids))
recognizer.save("face-trainner.yml")
