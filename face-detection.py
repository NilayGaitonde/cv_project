import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2.data
import os 
import torch
from embeddings import load_model, figureout_chromadb
import chromadb
from attendance import add_attendance, initialize
import pandas as pd
from backend import load_student_embeddings, recognize_faces_in_frame
import argparse
# from deepface import DeepFace

def search_embedding(device:str, chromadb_client, collection_name:str, model_path:str,data_path:str, img: np.ndarray):
    client = chromadb.PersistentClient(path=data_path)
    siamese = load_model(model_path,device)
    img = cv2.resize(img, (224, 224))
    img = torch.FloatTensor(img).permute(2,0,1)
    img = img.to(device)

    collection = client.get_collection(collection_name)

    with torch.no_grad():
        embedding = siamese.forward_one(img.unsqueeze(0))
        embedding = embedding.cpu().numpy()
        results = collection.query(query_embeddings=embedding, n_results=5)
    return results["ids"][0]


def face_detection(video:bool=True, img_path:str=None):
    best_score,best_result = 0,None
    initial_time, df = initialize()
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    student_embeddings = load_student_embeddings("/Users/nilaygaitonde/Documents/Projects/cv_project/students/ni")
    # face_cascade = cv2.CascadeClassifier(cascade_path)
    if video:
        try:
            cap = cv2.VideoCapture(1)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame, results, faces = recognize_faces_in_frame(frame, student_embeddings)
                for i,face in enumerate(faces):
                    try:
                        print(results)
                        df = add_attendance(results[i],df)
                        # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
                        # df = add_attendance(results[face],df)
                        # print(f"SAP ID: {results[0][0]} distance: {round(results[0][1],5)}")
                        # cv2.putText(frame,str(i),(face.left()-10,face.top()),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                        # cv2.putText(frame,f"SAP ID: {results[0][0]} distance: {round(results[0][1],2)}",(face.left()-10,face.top()),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                    except IndexError:
                        pass
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    df.to_csv(f"attendance_{initial_time}.csv",index=False)
                    break
        except KeyboardInterrupt:
            cap.release()
            cv2.destroyAllWindows()
            df.to_csv(f"attendance_{initial_time}.csv",index=False)
    else:
        if img_path is not None:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256,256))
            frame, results, faces = recognize_faces_in_frame(img, student_embeddings)
            print(results, faces)
            for i,face in enumerate(faces):
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
                cv2.putText(frame,str(i),(face.left()-10,face.top()),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                # print(f"SAP ID: {results[0][0]} distance: {round(results[0][1],5)}")
                # cv2.putText(frame,f"SAP ID: {results[0][0]} distance: {round(results[0][1],2)}",(faces[0].left()-10,faces[0].top()),cv2.FONT_HERSHEY_SIMPLEX,0.1,(255,255,255),2)
                cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection using embeddings")
    parser.add_argument("--video",action="store_true",help="Use this flag to enable video search")
    parser.add_argument("--img",type=str,help="Path to the image")
    args = parser.parse_args()
    if args.video:
        face_detection()
    elif args.img:
        face_detection(video=False, img_path=args.img)
    else:
        print("Please provide the correct arguments")
        exit(1)
    
    # how do i use this when running the python file
    # python face-detection.py --img 
    # face_detection()