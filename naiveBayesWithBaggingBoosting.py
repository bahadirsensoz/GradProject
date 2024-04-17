from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
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
X = dataset.transform(dataset.image_files)
y = dataset.labels()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes classifier
base_naive_bayes = GaussianNB()

# bagging
bagging_naive_bayes = BaggingClassifier(GaussianNB(), n_estimators=10, random_state=42)
bagging_naive_bayes.fit(X_train, y_train)
y_pred_bagging = bagging_naive_bayes.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging) * 100
print(f'Accuracy on test set (Bagging Naive Bayes): {accuracy_bagging:.2f}%')

# boosting
boosting_naive_bayes = AdaBoostClassifier(GaussianNB(), n_estimators=10, algorithm='SAMME', random_state=42) #"FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6.
                                                                                          #Use the SAMME algorithm to circumvent this warning" uyarısı geliyordu, algorithm='SAMME' ekleyerek çözülüyor.
boosting_naive_bayes.fit(X_train, y_train)
y_pred_boosting = boosting_naive_bayes.predict(X_test)
accuracy_boosting = accuracy_score(y_test, y_pred_boosting) * 100
print(f'Accuracy on test set (Boosting Naive Bayes): {accuracy_boosting:.2f}%')

# cross-validation with bagging
bagging_cv_scores = cross_val_score(bagging_naive_bayes, X, y, cv=5)
print("Bagging Cross-Validation Scores:", bagging_cv_scores)
print("Bagging Mean Accuracy:", np.mean(bagging_cv_scores))

# cross-validation with boosting
boosting_cv_scores = cross_val_score(boosting_naive_bayes, X, y, cv=5)
print("Boosting Cross-Validation Scores:", boosting_cv_scores)
print("Boosting Mean Accuracy:", np.mean(boosting_cv_scores))
