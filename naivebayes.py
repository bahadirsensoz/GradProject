import time
import psutil
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import glob
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class MammographyDataset:
    def __init__(self, data_dir, data_transform=None):
        self.data_dir = data_dir
        self.data_transform = data_transform
        self.image_files = glob.glob(os.path.join(data_dir, '*.jpg'))
        np.random.shuffle(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def transform(self, image_files):
        transformed_images = []
        for img_name in image_files:
            image = Image.open(img_name)
            if self.data_transform:
                image = self.data_transform(image)
            transformed_images.append(np.array(image).flatten())
        return np.array(transformed_images)

    def labels(self):
        return [int(os.path.basename(img_name).split('-')[0]) - 1 for img_name in self.image_files]


dataset = MammographyDataset(data_dir='D:/jpeg', data_transform=transform)

print(f'Calculations started...')
counter_start = time.time()  #Counter starts here for calculating total runtime, as well as cpu and memory trackers
cpu_start = psutil.cpu_percent(interval=1)
memory_start = psutil.virtual_memory()

X = dataset.transform(dataset.image_files)
y = dataset.labels()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train Naive Bayes classifier
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# test Naive Bayes classifier
y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb) * 100
print(f'Accuracy on test set (Naive Bayes): {accuracy_nb:.2f}%')

# cross-validation
cv_scores = cross_val_score(naive_bayes, X, y, cv=5)  # 5-fold cross-validation
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))

counter_end = time.time()  #Program ends here
cpu_end = psutil.cpu_percent(interval=1) #Set to 1 second, can be adjusted if we want
memory_end = psutil.virtual_memory() #Printing the total runtime along with final cpu and memory usages
print(f"Runtime of the program is {counter_end - counter_start:.2f} seconds.")
print(f'Initial CPU usage: {cpu_start}%, Final CPU usage: {cpu_end}%')
print(f'Initial Memory usage: {memory_start.percent}%, Final Memory usage: {memory_end.percent}%')
