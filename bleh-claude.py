import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torchvision
import torch
from PIL import Image,ImageEnhance

files = os.listdir("/Users/nilaygaitonde/Documents/Projects/cv_project/students")

for file in files:
    if file in ["ni","li",".DS_Store"]:
        continue
    img = cv2.imread(f"/Users/nilaygaitonde/Documents/Projects/cv_project/students/{file}")
    print(img.shape)
    if img.shape != (256,256,3):
        print(f"Resizing {file}")
        img = cv2.resize(img, (256,256))
        print(img.shape,"\n======")
    img = Image.fromarray(img)
    file = file.split(".")[0]
    brightened_image = ImageEnhance.Brightness(img).enhance(0.9)
    try:
        brightened_image.save(f"/Users/nilaygaitonde/Documents/Projects/cv_project/students/ni/{file}_1.jpg")
    except FileNotFoundError:
        # os.makedirs("/Users/nilaygaitonde/Documents/Projects/cv_project/students/ni")
        brightened_image.save(f"/Users/nilaygaitonde/Documents/Projects/cv_project/students/ni/{file}_1.jpg")