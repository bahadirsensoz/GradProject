import time
import numpy as np
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score
from statistics import harmonic_mean
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#checking for gpu if it is avaiable to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# needed transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class MammographyDataset(Dataset):
    def __init__(self, data_dir, data_transform=None):
        self.data_dir = data_dir
        self.data_transform = data_transform
        self.image_files = glob.glob(os.path.join(data_dir, '*.jpg'))
        np.random.shuffle(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        label = int(os.path.basename(img_name).split('-')[0]) - 1
        if self.data_transform:
            image = self.data_transform(image)
        return image, label


dataset = MammographyDataset(data_dir='D:/jpeg', data_transform=transform)

# split dataset
X = [dataset[i][0].numpy() for i in range(len(dataset))]
y = [dataset[i][1] for i in range(len(dataset))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# random forest 
print("Training Random Forest Classifier...")
vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
vit_model = vit_model.to(device) #since we are utilizing GPU now, moving the model to the GPU is required 
vit_features = [vit_model(torch.tensor(img).unsqueeze(0).to(device)).cpu().detach().numpy().flatten() for img in X_train]
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(vit_features, y_train)
vit_features_test = [vit_model(torch.tensor(img).unsqueeze(0).to(device)).cpu().detach().numpy().flatten() for img in X_test]
y_pred_rf = rf_model.predict(vit_features_test)
print("Random Forest Classifier trained.")

# naive bayes 
print("Training Naive Bayes Classifier...")
nb_model = GaussianNB()
X_train_nb = [img.flatten() for img in X_train]
X_test_nb = [img.flatten() for img in X_test]
nb_model.fit(X_train_nb, y_train)
y_pred_nb = nb_model.predict(X_test_nb)
print("Naive Bayes Classifier trained.")

# CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 224 * 224, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("Training CNN Model...")
cnn_model = CNNModel().to(device) #since we are utilizing GPU now, moving the model to the GPU is required 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=16, shuffle=True)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=16, shuffle=False)

num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = torch.tensor(images).float().to(device) #since we are utilizing GPU now, moving the model to the GPU is required 
        labels = torch.tensor(labels).long().to(device)

        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
print("CNN Model trained.")

# test CNN 
y_pred_cnn = []
with torch.no_grad():
    for images, labels in test_loader:
        images = torch.tensor(images).float().to(device) #since we are utilizing GPU now, moving the model to the GPU is required 
        labels = torch.tensor(labels).long().to(device)

        outputs = cnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred_cnn.extend(predicted.cpu().numpy())

# majority voting part
print("Performing Majority Voting...")
y_pred_ensemble = []
for i in range(len(y_test)):
    votes = [y_pred_rf[i], y_pred_nb[i], y_pred_cnn[i]]
    y_pred_ensemble.append(np.bincount(votes).argmax())

# evaluate majority voting
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble) * 100
print(f'Accuracy on test set (Majority Voting): {accuracy_ensemble:.2f}%')
