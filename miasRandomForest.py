import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# Define the dataset class
class MIASDataset:
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
        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        image = rgb2gray(image)  # Convert image to grayscale
        hog_features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        label = self.img_labels.iloc[idx, 3]
        return hog_features, label

# Define transformations
def transform_image(image):
    image = np.array(image)  # Convert PIL Image to NumPy array
    image = resize(image, (224, 224))  # Resize the image
    return image

# Create the dataset
dataset = MIASDataset(annotations_file='D:/mias_metadata.txt', img_dir='D:/all-mias', transform=transform_image)

# Extract features and labels
features = []
labels = []

for i in tqdm(range(len(dataset))):
    feature, label = dataset[i]
    features.append(feature)
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train the Random Forest classifier
rf_classifier = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
rf_classifier.fit(X_train_balanced, y_train_balanced)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
jaccard = jaccard_score(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%, Specificity: {specificity*100:.2f}%, Sensitivity: {sensitivity*100:.2f}%, Jaccard Index: {jaccard*100:.2f}%")
