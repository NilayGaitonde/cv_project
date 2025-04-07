import torch
import torchvision
from torch.nn import functional as F
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torch import nn
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import wandb
import pandas as pd


def create_pairs_from_triplets(triplet_data):
    """
    Convert triplet data (anchor, positive, negative) to pairs with labels. 
    1 for dissimilar pairs and 0 for similar pairs.
    """
    pairs_data = []
    
    for i in range(len(triplet_data)):
        anchor = triplet_data.iloc[i, 0]
        positive = triplet_data.iloc[i, 2]
        negative = triplet_data.iloc[i, 4]
        
        pairs_data.append([anchor, positive, 0])
        
        pairs_data.append([anchor, negative, 1])
    
    return pd.DataFrame(pairs_data, columns=['img1', 'img2', 'label'])


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, root_dir):
        self.data = data
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1 = plt.imread(os.path.join(self.root_dir, self.data.iloc[idx, 0]))
        img1 = cv2.resize(img1, (224, 224))
        img1 = img1.astype(np.float32) / 255.0
        img1 = torch.FloatTensor(img1).permute(2,0,1)
        img2 = plt.imread(os.path.join(self.root_dir, self.data.iloc[idx, 1]))
        img2 = cv2.resize(img2, (224, 224))
        img2 = img2.astype(np.float32) / 255.0
        img2 = torch.FloatTensor(img2).permute(2,0,1)
        label = self.data.iloc[idx, 2]
        return img1, img2, torch.FloatTensor([label])

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseNetwork, self).__init__()
        
        # Use a pretrained model as the base encoder
        # MobileNetV2 is lightweight and works well on M1 Macs
        base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        
        # Remove the classifier (last layer)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # Feature dimension from MobileNetV2
        feature_dim = 1280
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_one(self, x):
        features = self.feature_extractor(x)
        embedding = self.embedding(features)
        # Normalize embedding to unit length
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss


def training(model, train_loader, val_loader, epochs=10, lr=0.001, patience=5, min_delta=0):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss()
    best_val_loss = np.inf
    training_loss = []
    validation_loss = []
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(epochs):
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model.train()
        running_loss = 0.0
        print(f"Training epoch {epoch+1}/{epochs}")
        for i, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Iteration {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            wandb.log({"training_loss": loss.item()})
            if i % 100 == 0:
                accuracy = calculate_accuracy(model,val_loader)
                wandb.log({"Training accuracy": accuracy})

        epoch_loss = running_loss / len(train_loader)
        training_loss.append(epoch_loss)
        print(f"Validating epoch {epoch+1}/{epochs}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (img1, img2, label) in enumerate(val_loader):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()
                if i % 100 == 0:
                    accuracy = calculate_accuracy(model,val_loader)
                    wandb.log({"Validation accuracy": accuracy})
        val_loss /= len(val_loader)
        validation_loss.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
        wandb.log({"validation_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoint.pth')
            print(f"Model saved with validation loss: {best_val_loss:.4f}")

        early_stopping(val_loss)

    return model, training_loss, validation_loss

def calculate_accuracy(model, val_loader, threshold=0.5):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(val_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Get embeddings
            output1, output2 = model(img1, img2)
            
            # Calculate distance
            distances = F.pairwise_distance(output1, output2)
            
            # Make predictions (0 = same, 1 = different)
            # If distance < threshold → same identity → prediction = 0
            # If distance >= threshold → different identity → prediction = 1
            predictions = (distances >= threshold).float()
            
            # Count correct predictions
            # In your data: label 0 = same, label 1 = different
            batch_correct = (predictions == label.squeeze()).sum().item()
            
            correct_count += batch_correct
            total_count += label.size(0)
            
            # Debug prints
            if i%100 == 0:
                print(f"Batch accuracy: {batch_correct}/{label.size(0)} = {batch_correct/label.size(0):.4f} Iteration {i+1}/{len(val_loader)}")
                wandb.log({"Batch accuracy": batch_correct/label.size(0)})
    
    final_accuracy = correct_count / total_count
    print(f"Final accuracy: {correct_count}/{total_count} = {final_accuracy:.4f} or {final_accuracy*100:.2f}%")
    wandb.log({"Accuracy": final_accuracy})
    
    return final_accuracy


def inference(model, img1, img2):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        img1, img2 = img1.to(device), img2.to(device)
        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2)
    
    return distance.item()

# wandb.login()
# run = wandb.init(
#     project="siamese-network",
#     config={
#         "epochs": 10,
#         "lr": 0.001,
#         "patience": 5,
#         "min_delta": 0,
#         "batch_size": 20
#     }
# )

if __name__ == "__main__":
    data_path = '/Users/nilaygaitonde/Documents/Projects/cv_project/celeba-face-recognition-triplets/CelebA FR Triplets/CelebA FR Triplets/'
    images_folder = os.path.join(data_path, 'images')
    train_file = "celeba-face-recognition-triplets/CelebA FR Triplets/CelebA FR Triplets/triplets.csv"

    data = pd.read_csv(train_file)
    train_data = data[:int(data.shape[0]*0.6)]
    test_data = data[int(data.shape[0]*0.6):]
    train_data = create_pairs_from_triplets(train_data)

    train_dataset = Dataset(train_data,"/Users/nilaygaitonde/Documents/Projects/cv_project/celeba-face-recognition-triplets/CelebA FR Triplets/CelebA FR Triplets/images")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)

    val_dataset = Dataset(train_data,"/Users/nilaygaitonde/Documents/Projects/cv_project/celeba-face-recognition-triplets/CelebA FR Triplets/CelebA FR Triplets/images")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=True)

    # trained_model = torch.load("best_model.pth")
    # load best_model.pth
    trained_model = SiameseNetwork(input_size=(224, 224), hidden_size=4096)
    trained_model.load_state_dict(torch.load("best_model.pth"))
    trained_model, training_loss, validation_loss = training(trained_model, train_dataloader, val_dataloader, epochs=10, lr=0.001)
    # accuracy = calculate_accuracy(trained_model, val_dataloader)
    # print(accuracy)

    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()