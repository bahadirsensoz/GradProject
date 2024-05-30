import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from tqdm import tqdm


# Define the dataset 
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


# define transformations with data augmentation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# dataset
dataset = MIASDataset(annotations_file='D:/mias_metadata.txt', img_dir='D:/all-mias', transform=transform)

# initialize the stratified k-fold cross-validation
k_folds = 5
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# training settings
num_epochs = 20
initial_learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_accuracy = 0.0

# cross-validation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, dataset.img_labels['SEVERITY'])):
    print(f'FOLD {fold + 1}')
    print('--------------------------------')

    # sample elements randomly from given list of ids
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # define data loaders
    train_loader = DataLoader(dataset, batch_size=16, sampler=train_subsampler)
    test_loader = DataLoader(dataset, batch_size=16, sampler=test_subsampler)

    # load pre-trained ViT model and freeze the backbone layers initially
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2,
                                                      ignore_mismatched_sizes=True)
    for param in model.vit.parameters():
        param.requires_grad = False

    model.to(device)

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # early stopping 
    early_stopping_patience = 3
    epochs_no_improve = 0

    # training 
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

        scheduler.step(running_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

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
        print(f"Validation Accuracy: {accuracy:.4f}")

        # check early stopping condition
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print("Early stopping triggered")
            break

    # unfreeze the backbone layers for fine-tuning after initial training
    for param in model.vit.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate / 10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

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

        scheduler.step(running_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Fine-Tuning Loss: {running_loss / len(train_loader):.4f}")

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
        print(f"Validation Accuracy: {accuracy:.4f}")

        # check early stopping condition
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print("Early stopping triggered")
            break

# final evaluation on the test set of the best model
model.load_state_dict(torch.load(f'best_model_fold_{np.argmax([best_accuracy])}.pth'))
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

print(
    f"Final Metrics - Accuracy: {accuracy * 100:.2f}%, Specificity: {specificity * 100:.2f}%, Sensitivity: {sensitivity * 100:.2f}%, Jaccard Index: {jaccard * 100:.2f}%")
