import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # Input is grayscale (1 channel)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the output dimensions of CNN
        # For 224x224 input, after convolutions and pooling:
        # First conv: 224-10+1 = 215, after pooling: 107
        # Second conv: 107-7+1 = 101, after pooling: 50
        # Third conv: 50-4+1 = 47, after pooling: 23
        # Fourth conv: 23-4+1 = 20
        # So output is 256 channels of 20x20 feature maps
        self.fc_input_dim = 256 * 20 * 20
        
        # Fully connected layer for embedding
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 4096),
            nn.Sigmoid()
        )
        
    def forward_one(self, x):
        # Get the features from CNN
        x = self.cnn(x)
        # Pass through fully connected layer for embedding
        x = self.fc(x)
        return x
    
    def forward(self, input1, input2):
        # Get embeddings for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        # Return both embeddings
        return output1, output2

# Contrastive Loss 
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        # Calculate Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Contrastive loss
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss

# Dataset class for pairs
class SiamesePairDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image paths
        img1_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        img2_path = os.path.join(self.root_dir, self.data.iloc[idx, 1])
        label = self.data.iloc[idx, 2]
        
        # Read images
        img1 = Image.open(img1_path).convert('L')  # Convert to grayscale
        img2 = Image.open(img2_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.FloatTensor([label])

# Function to create pairs from triplet data
def create_pairs_from_triplets(triplet_data):
    """
    Convert triplet data (anchor, positive, negative) to pairs with labels
    """
    pairs_data = []
    
    # For each triplet
    for i in range(len(triplet_data)):
        anchor = triplet_data.iloc[i, 0]
        positive = triplet_data.iloc[i, 2]
        negative = triplet_data.iloc[i, 4]
        
        # Create positive pair (anchor, positive) with label 1
        pairs_data.append([anchor, positive, 0])  # 0 for same identity
        
        # Create negative pair (anchor, negative) with label 0
        pairs_data.append([anchor, negative, 1])  # 1 for different identity
    
    return pd.DataFrame(pairs_data, columns=['img1', 'img2', 'label'])

# Training function
def train_siamese_network(model, train_loader, val_loader, epochs=20, lr=0.0005):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Training
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (img1, img2, label) in progress_bar:
            # Move data to device
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output1, output2 = model(img1, img2)
            
            # Calculate loss
            loss = criterion(output1, output2, label)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        training_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        validation_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {epoch_loss:.4f} | Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_siamese_model.pth')
            print(f"Model saved with validation loss: {best_val_loss:.4f}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), training_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.show()
    
    return model

# Function to calculate accuracy on validation set
def calculate_accuracy(model, val_loader, threshold=0.5):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Get embeddings
            output1, output2 = model(img1, img2)
            
            # Calculate distance
            euclidean_distance = F.pairwise_distance(output1, output2)
            
            # If distance is less than threshold, predict same identity
            predictions = (euclidean_distance < threshold).float()
            
            # Calculate correct predictions (need to invert since label 0 means same identity in our dataset)
            correct += ((predictions == (1 - label)).sum().item())
            total += label.size(0)
    
    accuracy = correct / total
    return accuracy

# Function to build face database for attendance system
def build_face_database(model, images_folder, student_ids):
    """
    Create a database of face embeddings for all students
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    embeddings_db = {}
    
    for student_id in student_ids:
        student_folder = os.path.join(images_folder, student_id)
        
        # Skip if not a directory
        if not os.path.isdir(student_folder):
            continue
        
        # Process all images for this student
        embeddings = []
        
        for img_name in os.listdir(student_folder):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(student_folder, img_name)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
                
                # Get embedding
                with torch.no_grad():
                    embedding = model.forward_one(img)
                
                embeddings.append(embedding.cpu().numpy())
        
        # Average all embeddings for this student
        if embeddings:
            avg_embedding = np.mean(np.vstack(embeddings), axis=0)
            embeddings_db[student_id] = avg_embedding
    
    return embeddings_db

# Function to recognize a face for attendance
def recognize_face(model, face_img, embeddings_db, threshold=0.5):
    """
    Recognize a face by comparing with database
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Preprocess image
    if isinstance(face_img, str):  # If path is provided
        face_img = Image.open(face_img).convert('L')
    else:  # If numpy array from camera
        face_img = Image.fromarray(face_img).convert('L')
    
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    
    # Get embedding
    with torch.no_grad():
        query_embedding = model.forward_one(face_tensor).cpu().numpy()
    
    # Compare with database
    best_match = None
    min_distance = float('inf')
    
    for student_id, db_embedding in embeddings_db.items():
        # Calculate Euclidean distance
        distance = np.linalg.norm(query_embedding - db_embedding)
        
        if distance < min_distance:
            min_distance = distance
            best_match = student_id
    
    # Return match if distance is below threshold
    if min_distance < threshold:
        return best_match, min_distance
    else:
        return None, min_distance

# Main execution
def main():
    # Paths and settings
    data_path = '/Users/nilaygaitonde/Documents/Projects/capstone/celeba-face-recognition-triplets/CelebA FR Triplets/CelebA FR Triplets/'
    images_folder = os.path.join(data_path, 'images')
    train_file = "celeba-face-recognition-triplets/CelebA FR Triplets/CelebA FR Triplets/triplets.csv"
    
    # Load triplet data
    train_data = pd.read_csv(train_file)
    
    # Convert triplets to pairs
    pairs_data = create_pairs_from_triplets(train_data)
    
    # Split into train and validation
    train_df, val_df = train_test_split(pairs_data, test_size=0.2, random_state=42)
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = SiamesePairDataset(train_df, images_folder, transform=transform)
    val_dataset = SiamesePairDataset(val_df, images_folder, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Create and train model
    model = SiameseNetwork()
    trained_model = train_siamese_network(model, train_loader, val_loader, epochs=20)
    
    # Calculate accuracy
    threshold = 0.5  # Adjust this based on validation
    accuracy = calculate_accuracy(trained_model, val_loader, threshold)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Build face database from your peer ID photos
    # Assume peer photos are organized in folders with student IDs
    peer_photos_folder = '/path/to/peer/photos'
    student_ids = [folder for folder in os.listdir(peer_photos_folder) 
                  if os.path.isdir(os.path.join(peer_photos_folder, folder))]
    
    embeddings_db = build_face_database(trained_model, peer_photos_folder, student_ids)
    
    # Save embeddings database for future use
    np.save('face_embeddings.npy', embeddings_db)
    
    # Example: Recognize a face
    test_face = '/path/to/test/face.jpg'
    student_id, distance = recognize_face(trained_model, test_face, embeddings_db)
    
    if student_id:
        print(f"Recognized as: {student_id} with confidence: {1-distance:.4f}")
    else:
        print("Face not recognized")

if __name__ == '__main__':
    main()