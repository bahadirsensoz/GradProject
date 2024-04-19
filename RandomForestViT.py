import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torchvision import models
import torchvision.transforms as transforms
import glob
import os
from PIL import Image
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Define the Visual Transformer (ViT) model
vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class MammographyDataset:
    def __init__(self, data_dir, data_transform=None):
        self.data_dir = data_dir
        self.data_transform = data_transform
        self.image_files = glob.glob(os.path.join(data_dir, '*.jpg'))
        print(f"Found {len(self.image_files)} image files in {data_dir}")
        np.random.shuffle(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def transform(self, image_files):
        transformed_images = []
        total_images = len(image_files)
        for i, img_name in enumerate(image_files):
            image = Image.open(img_name)
            if self.data_transform:
                image = self.data_transform(image)
            transformed_images.append(image.numpy())
            print(f'Processed image {i+1}/{total_images} ({(i+1)/total_images*100:.2f}%)', end='\r')
        return transformed_images

    def labels(self):
        labels = [int(os.path.basename(img_name).split('-')[0]) - 1 for img_name in self.image_files]
        print(f"\nGenerated {len(labels)} labels")
        return labels

dataset = MammographyDataset(data_dir='/Users/bahadir/Desktop/GradProject/jpeg', data_transform=transform)

print(f"Calculations started...")
counter_start = time.time()  # Counter starts here for calculating total runtime

X = np.array(dataset.transform(dataset.image_files))
y = dataset.labels()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract features using the Visual Transformer (ViT) model
vit_features = []
total_images = len(X_train)
for i, img in enumerate(X_train):
    img = torch.tensor(img).unsqueeze(0)
    features = vit_model(img)
    vit_features.append(features.detach().numpy().flatten())
    print(f'Extracted features from image {i+1}/{total_images} ({(i+1)/total_images*100:.2f}%)', end='\r')
print()

vit_features = np.array(vit_features)

# Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(vit_features, y_train)

print("Evaluation on the test set:")

# Extract features for test set
vit_features_test = []
total_images_test = len(X_test)
for i, img in enumerate(X_test):
    img = torch.tensor(img).unsqueeze(0)
    features = vit_model(img)
    vit_features_test.append(features.detach().numpy().flatten())
    print(f'Extracted features from test image {i+1}/{total_images_test} ({(i+1)/total_images_test*100:.2f}%)', end='\r')
print()

vit_features_test = np.array(vit_features_test)

# Predictions on the test set
y_pred_rf = rf_model.predict(vit_features_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
print(f'Accuracy on test set (Random Forest): {accuracy_rf:.2f}%')

print("Total runtime:", time.time() - counter_start)
