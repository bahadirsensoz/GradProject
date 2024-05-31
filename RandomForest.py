import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# Load metadata
df_meta = pd.read_csv('F:/Dataset/archive/csv/meta.csv')
df_dicom = pd.read_csv('F:/Dataset/archive/csv/dicom_info.csv', delimiter=';')

cropped_images = df_dicom[df_dicom.SeriesDescription == 'cropped images'].image_path
full_mammo = df_dicom[df_dicom.SeriesDescription == 'full mammogram images'].image_path
roi_img = df_dicom[df_dicom.SeriesDescription == 'ROI mask images'].image_path

imdir = 'F:/Dataset/archive/jpeg'
cropped_images = cropped_images.replace('CBIS-DDSM/jpeg', imdir, regex=True)
full_mammo = full_mammo.replace('CBIS-DDSM/jpeg', imdir, regex=True)
roi_img = roi_img.replace('CBIS-DDSM/jpeg', imdir, regex=True)

# Organize image paths
full_mammo_dict = {dicom.split("/")[4]: dicom for dicom in full_mammo}
cropped_images_dict = {dicom.split("/")[4]: dicom for dicom in cropped_images}
roi_img_dict = {dicom.split("/")[4]: dicom for dicom in roi_img}

# Load the mass dataset
mass_train = pd.read_csv('F:/Dataset/archive/csv/mass_case_description_train_set.csv')
mass_test = pd.read_csv('F:/Dataset/archive/csv/mass_case_description_test_set.csv')

def fix_image_path(data):
    for index, img in enumerate(data.values):
        img_name = img[11].split("/")[2]
        data.iloc[index, 11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        data.iloc[index, 12] = cropped_images_dict[img_name]

fix_image_path(mass_train)
fix_image_path(mass_test)

mass_train = mass_train.rename(columns={
    'left or right breast': 'left_or_right_breast',
    'image view': 'image_view',
    'abnormality id': 'abnormality_id',
    'abnormality type': 'abnormality_type',
    'mass shape': 'mass_shape',
    'mass margins': 'mass_margins',
    'image file path': 'image_file_path',
    'cropped image file path': 'cropped_image_file_path',
    'ROI mask file path': 'ROI_mask_file_path'
})

mass_test = mass_test.rename(columns={
    'left or right breast': 'left_or_right_breast',
    'image view': 'image_view',
    'abnormality id': 'abnormality_id',
    'abnormality type': 'abnormality_type',
    'mass shape': 'mass_shape',
    'mass margins': 'mass_margins',
    'image file path': 'image_file_path',
    'cropped image file path': 'cropped_image_file_path',
    'ROI mask file path': 'ROI_mask_file_path'
})

mass_train['mass_shape'] = mass_train['mass_shape'].fillna(method='bfill')
mass_train['mass_margins'] = mass_train['mass_margins'].fillna(method='bfill')
mass_test['mass_margins'] = mass_test['mass_margins'].fillna(method='bfill')

full_mass = pd.concat([mass_train, mass_test], axis=0)

# Image preprocessing
def image_processor(image_path, target_size):
    absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(absolute_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image_array = image / 255.0
    return image_array

# Preprocess and prepare dataset
def preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for image_path in image_paths:
        image = image_processor(image_path, target_size)
        images.append(image)
    return np.array(images)

X = preprocess_images(full_mass['image_file_path'])
y = full_mass['pathology'].replace({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}).values

# Train test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data_augmented = train_datagen.flow(X_train, y_train, batch_size=32)
val_data_augmented = val_datagen.flow(X_val, y_val, batch_size=32)
test_data_augmented = val_datagen.flow(X_test, y_test, batch_size=32)

# Feature extraction using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(data_gen, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))
    labels = np.zeros(shape=(sample_count))
    i = 0
    for inputs_batch, labels_batch in data_gen:
        features_batch = base_model.predict(inputs_batch)
        features[i * data_gen.batch_size : (i + 1) * data_gen.batch_size] = features_batch
        labels[i * data_gen.batch_size : (i + 1) * data_gen.batch_size] = labels_batch
        i += 1
        if i * data_gen.batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_data_augmented, len(X_train))
val_features, val_labels = extract_features(val_data_augmented, len(X_val))
test_features, test_labels = extract_features(test_data_augmented, len(X_test))

# Flatten the features
train_features = np.reshape(train_features, (len(X_train), 7 * 7 * 512))
val_features = np.reshape(val_features, (len(X_val), 7 * 7 * 512))
test_features = np.reshape(test_features, (len(X_test), 7 * 7 * 512))

# Train a RandomForest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_features, train_labels)

# Evaluate the model
y_pred_train = rf.predict(train_features)
y_pred_test = rf.predict(test_features)

train_cm = confusion_matrix(train_labels, y_pred_train)
test_cm = confusion_matrix(test_labels, y_pred_test)

# Calculate additional metrics
def calculate_metrics(cm, y_true, y_pred):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    jaccard = jaccard_score(y_true, y_pred)
    return accuracy, sensitivity, specificity, precision, jaccard

train_accuracy, train_sensitivity, train_specificity, train_precision, train_jaccard = calculate_metrics(train_cm, train_labels, y_pred_train)
test_accuracy, test_sensitivity, test_specificity, test_precision, test_jaccard = calculate_metrics(test_cm, test_labels, y_pred_test)

print(f"Train Set Metrics:\n Accuracy: {train_accuracy:.4f}\n Sensitivity: {train_sensitivity:.4f}\n Specificity: {train_specificity:.4f}\n Precision: {train_precision:.4f}\n Jaccard Index: {train_jaccard:.4f}\n")
print(f"Test Set Metrics:\n Accuracy: {test_accuracy:.4f}\n Sensitivity: {test_sensitivity:.4f}\n Specificity: {test_specificity:.4f}\n Precision: {test_precision:.4f}\n Jaccard Index: {test_jaccard:.4f}\n")

# Plot confusion matrices
def plot_confusion_matrix(cm, labels, title):
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized_cm = cm / row_sums

    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_cm, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

cm_labels = ['MALIGNANT', 'BENIGN']
print(f"Train Set Classification report:\n {classification_report(train_labels, y_pred_train, target_names=cm_labels)}\n")
plot_confusion_matrix(train_cm, cm_labels, 'Train Set Confusion Matrix')

print(f"Test Set Classification report:\n {classification_report(test_labels, y_pred_test, target_names=cm_labels)}\n")
plot_confusion_matrix(test_cm, cm_labels, 'Test Set Confusion Matrix')
