import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from tqdm import tqdm

# the dataset class
class MIASDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, delim_whitespace=True)
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = self.img_labels.dropna(subset=['SEVERITY'])
        self.img_labels = self.img_labels[self.img_labels['SEVERITY'].isin(['B', 'M'])]
        self.label_encoder = LabelEncoder()
        self.img_labels['SEVERITY'] = self.label_encoder.fit_transform(self.img_labels['SEVERITY'])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.img_labels.iloc[idx, 0]}.pgm")
        image = Image.open(img_name).convert("RGB")
        label = self.img_labels.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, label

# transformations with data augmentation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# dataset and dataloaders
dataset = MIASDataset(annotations_file='D:/mias_metadata.txt', img_dir='D:/all-mias', transform=transform)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# class weights for handling class imbalance
class_counts = np.bincount([label for _, label in dataset])
class_weights = 1. / class_counts
weights = [class_weights[label] for _, label in train_data]
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

# DataLoader to use weighted sampler
train_loader = DataLoader(train_data, batch_size=16, sampler=sampler)

# pre-trained ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# training loop
num_epochs = 10
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    jaccard = jaccard_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}, Jaccard Index: {jaccard:.4f}")

# load the best model for final evaluation
model.load_state_dict(torch.load('best_model.pth'))

# final evaluation on the test set
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).long()
        outputs = model(images).logits
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
jaccard = jaccard_score(all_labels, all_preds)

print(f"Final Metrics - Accuracy: {accuracy*100:.2f}%, Specificity: {specificity*100:.2f}%, Sensitivity: {sensitivity*100:.2f}%, Jaccard Index: {jaccard*100:.2f}%")
