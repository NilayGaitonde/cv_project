import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import dlib
from typing import List, Union, Tuple

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def load_student_embeddings(student_img_path:str) -> dict:
    student_encodings = {}

    for filename in os.listdir(student_img_path):
        sap_id = os.path.splitext(filename)[0]
        img_path = os.path.join(student_img_path, filename)
        img = plt.imread(img_path)
        faces = detector(img,1)
        if faces:
            face = faces[0]
            shape = sp(img, face)
            encoding = face_rec.compute_face_descriptor(img, shape)

            student_encodings[sap_id] = np.array(encoding)
        else:
            print(f"No face detected for {sap_id}")
    return student_encodings

def recognize_faces_in_frame(frame: np.ndarray, student_encodings: dict, tolerance: float=0.5) -> List[str]:
    faces = detector(frame,1)
    recognized_students = {}
    found_students = []

    for face in faces:
        recognized_students = {}
        shape = sp(frame, face)
        encoding = face_rec.compute_face_descriptor(frame, shape)
        for sap_id, student_encoding in student_encodings.items():
            distance = np.linalg.norm(encoding - student_encoding)
            if distance < tolerance:
                recognized_students[sap_id] = distance
        recognized_students = sorted(recognized_students.items(), key=lambda x: x[1])[::-1]
        if recognized_students:
            sap_id, distance = recognized_students[0]
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
            cv2.putText(frame, f"SAP ID: {sap_id} distance: {round(distance, 5)}", (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            found_students.append(sap_id)
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
            cv2.putText(frame, "Unknown", (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame, found_students, faces


if __name__ == "__main__":
    student_encodings = load_student_embeddings("/Users/nilaygaitonde/Documents/Projects/cv_project/students/")
    saps = list(student_encodings.keys())
    saps = sorted(saps)
    # print(saps)
    frame = cv2.imread("face.png")
    frame, recognized_students,faces = recognize_faces_in_frame(frame, student_encodings)
    print(recognized_students[0][0])