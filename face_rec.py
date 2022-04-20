# installing required libraries

import os

import cv2
import face_recognition
import face_recognition as fr
import numpy as np


# encodes the face-images
def get_encoded_faces():
    encoded_result = {}
    for dir_path, dir_names, face_names in os.walk("./faces"):
        for face in face_names:
            if face.endswith(".jpg") or face.endswith(".png"):
                face_img = fr.load_image_file("faces/" + face)
                encoding = fr.face_encodings(face_img)[0]
                encoded_result[face.split(".")[0]] = encoding
    return encoded_result


# finds the labels or face names
def classify_face(im):
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    img = cv2.imread(im, 1)
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 40), (255, 0, 0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(img, (left - 20, bottom - 5), (right + 20, bottom + 40), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left, bottom + 30), font, 1.0, (255, 255, 255), 2)
    # Display the resulting image
    while True:
        cv2.imshow('Face-image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names


print(classify_face("test/test.png"))
