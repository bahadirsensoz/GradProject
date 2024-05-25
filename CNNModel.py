import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import glob
import os
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class MammographyDataset(Dataset):
    def _init_(self, data_dir, data_transform=None):
        self.data_dir = data_dir
        self.data_transform = data_transform
        self.image_files = glob.glob(os.path.join(data_dir, '*.jpg'))
        # os.listdir kullanınca perm hatası veriyor, glob kullanıldı

    def _len_(self):
        return len(self.image_files)

    def _getitem_(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        # baştaki 1 ve 2 sayısını label olarak al
        label = int(os.path.basename(img_name).split('-')[0])
        label = label - 1  # labellar 0-1 olsun istiyoruz
        if self.data_transform:
            image = self.data_transform(image)
        return image, label


dataset = MammographyDataset(data_dir='D:/jpeg', data_transform=transform) #transform yukarıda

# train/test
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


# CNN modeli
class CNNModel(nn.Module):
    def _init_(self):
        super(CNNModel, self)._init_()
        # Define the layers of your CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Adjusted input size for fc1
        self.fc1 = nn.Linear(32 * 224 * 224, 1000)  # tensor'ün boyutuna göre ayarlandı
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # Flatten the output tensor (internetten bakıldı, hata çözümü için)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Initialize counters for calculating Se, Sp, Acc, and JI
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# model test
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate confusion matrix values
        for i in range(len(labels)):
            if labels[i] == 1 and predicted[i] == 1:
                true_positive += 1
            elif labels[i] == 0 and predicted[i] == 0:
                true_negative += 1
            elif labels[i] == 0 and predicted[i] == 1:
                false_positive += 1
            elif labels[i] == 1 and predicted[i] == 0:
                false_negative += 1

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')

# Calculate Se, Sp, and JI
sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)
jaccard_index = true_positive / (true_positive + false_positive + false_negative)

print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'Jaccard Index: {jaccard_index:.2f}')
