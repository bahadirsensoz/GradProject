import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import glob
import os
from PIL import Image
import time
import psutil

#GPU kullanma denemesi olarak (cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class MammographyDataset(Dataset):
    def __init__(self, data_dir, data_transform=None):
        self.data_dir = data_dir
        self.data_transform = data_transform
        self.image_files = glob.glob(os.path.join(data_dir, '*.jpg'))
        # os.listdir kullanınca perm hatası veriyor, glob kullanıldı
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        # baştaki 1 ve 2 sayısını label olarak al
        label = int(os.path.basename(img_name).split('-')[0])
        label = label - 1  # labellar 0-1 olsun istiyoruz
        if self.data_transform:
            image = self.data_transform(image)
        return image, label


dataset = MammographyDataset(data_dir="C:/Users/canok/OneDrive/Masaüstü/bitirme/jpeg", data_transform=transform) #transform yukarıda

print(f'Calculations started...')
#starting to record time just after declaring dataset as we did in naive
start_time = time.time()
cpu_start = psutil.cpu_percent(interval=1) #Same we used in naivebayes.py
memory_start = psutil.virtual_memory()

# train/test
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True) #batch size ilerki denemelerde degistirilebilir
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


# CNN modeli
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define the layers of your CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)  # out_channels azaltıldı, vram eksikti
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)  # 8gb vram yetersiz kaldı
        # Adjusted input size for fc1
        self.fc1 = nn.Linear(16 * 224 * 224, 500)  # Reduced output size # tensor'ün boyutuna göre ayarlandı
        self.fc2 = nn.Linear(500, 2)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # Flatten the output tensor (internetten bakıldı, hata çözümü için)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#model = CNNModel() to(device) olarak deneme (CUDA)
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) #CUDA DENEME
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# model test
correct = 0
total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  #CUDA DENEME
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy()) #When experimenting, I found that we have to use .cpu() to use the numpy functions, moving them to cpu to do calculations.
        all_predictions.extend(predicted.cpu().numpy()) #After doing that, we can use numpy function to store our values into arrays, that we will store in all_labels and all_predictions arrays.
        
accuracy = 100 * correct / total
precision = 100 * precision_score(all_labels, all_predictions, average='macro')
recall = 100 * recall_score(all_labels, all_predictions, average='macro')
f1 = 100 * f1_score(all_labels, all_predictions, average='macro')

print(f'Accuracy on test set: {accuracy:.2f}%')
print(f'Precision on test set: {precision:.2f}%')
print(f'Recall on test set: {recall:.2f}%')
print(f'F1 Score on test set: {f1:.2f}%')

#end of computations
end_time = time.time()
#printing total time here
total_runtime = end_time - start_time
cpu_end = psutil.cpu_percent(interval=1) #Set to 1 second, can be adjusted if we want
memory_end = psutil.virtual_memory()

print(f'Total runtime is: {total_runtime:.2f} seconds')
print(f'Initial CPU usage: {cpu_start}%, Final CPU usage: {cpu_end}%')
print(f'Initial Memory usage: {memory_start.percent}%, Final Memory usage: {memory_end.percent}%')
