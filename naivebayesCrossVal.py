from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
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
X = dataset.transform(dataset.image_files)
y = dataset.labels()


naive_bayes = GaussianNB()

# cross-validation
cv_scores = cross_val_score(naive_bayes, X, y, cv=5)  # 5 fold cross-validation
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
