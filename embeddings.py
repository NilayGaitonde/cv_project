import chromadb
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from torch import nn
from torch.nn import functional as F

from siamese import SiameseNetwork
from chromadb.errors import InvalidCollectionException

def get_image(image_path:str):
    return plt.imread(image_path)

def load_model(model_path:str,device:str):
    siamese = SiameseNetwork(input_size=(224, 224), hidden_size=4096)
    siamese.to(device)
    siamese.load_state_dict(torch.load(model_path))
    siamese.eval()
    return siamese


def figureout_chromadb(data_path:str, collection_name:str):
    client = chromadb.PersistentClient(path=data_path)
    try:
        collection = client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists.")
    except Exception as e:
        collection = client.create_collection(collection_name)
        print(f"Collection {collection_name} created.")
    return client, collection

def create_embeddings(embedding_path:str,image_root_path:str,model_path:str,device:str,chromadb_client:chromadb.api.client.Client,collection_name:str):
    siamese = load_model(model_path,device)
    sap_ids = os.listdir(image_root_path)
    collection = chromadb_client.get_collection(collection_name)
    for sap_id in sap_ids:
        image_path = os.path.join(image_root_path,sap_id,f"{sap_id}.JPG")
        image = get_image(image_path)
        image = cv2.resize(image, (224, 224))
        image = torch.FloatTensor(image).permute(2,0,1)

        with torch.no_grad():
            image = image.to(device)
            embedding = siamese.forward_one(image.unsqueeze(0))
            embedding = embedding.cpu().numpy()
            collection.add(embeddings=embedding,ids=[sap_id])
            print(f"Inserted {sap_id} into {collection_name}")

if __name__ == "__main__":
    data_path = "/Users/nilaygaitonde/Documents/Projects/cv_project/embeddings/"
    image_root_path = "/Users/nilaygaitonde/Documents/Projects/cv_project/data/"
    model_path = "best_model.pth"
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")

    client, collection = figureout_chromadb(data_path, "CV_Project")
    create_embeddings(data_path,image_root_path,model_path,device,client,"CV_Project")