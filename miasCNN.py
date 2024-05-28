import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

# metadata
metadata_path = 'D:/mias_metadata.txt'
images_path = 'D:/all-mias'

metadata = pd.read_csv(metadata_path, delim_whitespace=True, header=0)
metadata = metadata[metadata['SEVERITY'].isin(['B', 'M'])]

# labels
label_encoder = LabelEncoder()
metadata['SEVERITY'] = label_encoder.fit_transform(metadata['SEVERITY'])


class MiasDataset(Dataset):
    def __init__(self, metadata, images_path, transform=None):
        self.metadata = metadata
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, f"{self.metadata.iloc[idx, 0]}.pgm")
        image = Image.open(img_name).convert('L')
        label = self.metadata.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        return image, label


# transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# dataset and dataloaders
dataset = MiasDataset(metadata, images_path, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# define the CNN model with increased dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)  # Increased dropout rate
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


model = CNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train the model with validation
def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            labels = labels.view(-1, 1).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.view(-1, 1).float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}")

    return train_losses, val_losses


# train the model and get losses
train_losses, val_losses = train_model_with_validation(model, train_loader, val_loader, criterion, optimizer,
                                                       num_epochs=30)

# plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds).flatten()

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    jaccard = jaccard_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Specificity: {specificity * 100:.2f}%")
    print(f"Sensitivity: {sensitivity * 100:.2f}%")
    print(f"Jaccard Index: {jaccard * 100:.2f}%")


evaluate_model(model, test_loader)
