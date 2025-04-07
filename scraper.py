import os
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import selenium.webdriver as webdriver
import selenium.webdriver.common.keys as keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from PIL import Image
import time
import torch
from torchvision import transforms

def download_image(url:str,path:str,websession:webdriver.firefox.webdriver.WebDriver) -> None:
    image_element = WebDriverWait(websession,10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[4]/div[1]/header/nav/div/ul/li[10]/a/img')))
    print("Found image element")
    if url:
        cookies = websession.get_cookies()
        session = requests.Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'])
        response = session.get(url)
        if response.status_code == 200:
            try:
                with open(path, 'wb') as file:
                    print("Downloading image...")
                    file.write(response.content)
                    print("Downloaded image",path)
            except FileNotFoundError:
                os.makedirs(os.path.dirname(path))
                with open(path, 'wb') as file:
                    print("Downloading image...")
                    file.write(response.content)
                    print("Downloaded image",path)
    
def augment_image(image_path:str, output_path:str, sap_id:str, output_size:tuple=(128,128),output_count:int=3) -> None:
    image = plt.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmentations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(output_size, scale=(0.8, 1.0)),  # Random cropping
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.RandomRotation(degrees=10),  # Random rotation
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),  # Add Gaussian noise
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Clamp values to [0, 1]
        transforms.ToPILImage(),  # Convert back to PIL image
    ])
    augmented_images = []
    for i in range(output_count):
        augmented_image = augmentations(image)
        augmented_image = np.array(augmented_image)  # Convert to numpy array
        plt.imsave(output_path+"/"+sap_id+"_a_"+str(i)+".jpg", augmented_image)
    
def login(url:str, payload:dict) -> webdriver:
    websession = webdriver.Firefox()
    websession.get("https://www.google.com")
    websession.get(url)
    time.sleep(5)
    print("Opened the website")
    username_field = WebDriverWait(websession, 10).until(EC.presence_of_element_located((By.NAME, "username")))
    print("Found username")
    username_field.send_keys(payload["username"])
    print("Entered username")
    password_field = WebDriverWait(websession,10).until(EC.presence_of_element_located((By.NAME,"password")))
    print("Found password")
    password_field.send_keys(payload["password"])
    print("Entered password")
    websession.find_element(by=By.ID,value="userLogin").click()
    print("Clicked login")
    time.sleep(5)
    print("Done sleeping...")
    return websession
    
def main():
    # sap_ids = [f"703210190{i}" for i in range(1,61)]
    # sap ids go from 70321019001 to 70321019060
    sap_ids = [f"7032101900{i}" for i in range(1,10)]
    sap_ids_2 = [f"703210190{i}" for i in range(10,62)]
    sap_ids.extend(sap_ids_2)
    print(sap_ids)
    

    login_url = "https://portal.svkm.ac.in/usermgmt/login"
    payload = {
        "username": "70321019023",
        "password": "nm1ms@Nilay"
    }
    websession = login(login_url, payload)
    for sap_id in sap_ids:
        photo_url = f"https://portal.svkm.ac.in/MPSTME-NM-M/savedImages/{sap_id}.JPG"
        path = f"students/"
        download_image(photo_url, f"{path}/{sap_id}.jpg", websession)
        img = plt.imread(f"{path}/{sap_id}.jpg")
        img = cv2.resize(img, (256,256))
        plt.imsave(f"{path}/{sap_id}.jpg", img)
        # augment_image(f"{path}/{sap_id}.JPG", path, output_count=3,sap_id = sap_id)

if __name__ == "__main__":
    main()
    # sap_ids = [f"7032101900{i}" for i in range(1,10)]
    # sap_ids_2 = [f"703210190{i}" for i in range(10,62)]
    # sap_ids.extend(sap_ids_2)
    # for sap_id in sap_ids:
    #     path = f"data/{sap_id}"
    #     img = plt.imread(f"{path}/{sap_id}.jpg")
    #     img = cv2.resize(img, (256,256))
    #     plt.imsave(f"{path}/{sap_id}.jpg", img)