import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import os
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW

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

# Convert integer labels to one-hot encoded labels
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
y_val = to_categorical(y_val, 2)

# Data augmentation
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_data_augmented = train_datagen.flow(X_train, y_train, batch_size=32)

# Ensure the shape is correct after augmentation
def check_batch_shape(data_gen):
    for X_batch, y_batch in data_gen:
        print(f"Batch X shape: {X_batch.shape}")
        break

check_batch_shape(train_data_augmented)

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define distributed training strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Define a CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax', dtype='float32')
    ])

    model.compile(
        optimizer=AdamW(learning_rate=0.0001, weight_decay=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# Define the learning rate schedule function
def lr_schedule(epoch, lr):
    if epoch < 7:
        return lr
    else:
        return lr * tf.math.exp(-0.01)

# Implement early stopping and learning rate scheduler
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Train model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, lr_scheduler]
)

model.summary()

# Evaluate the model
model.evaluate(X_test, y_test)

# Classification reports and confusion matrix
cm_labels = ['MALIGNANT', 'BENIGN']

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

y_pred_classes_test = np.argmax(y_pred_test, axis=1)
y_pred_classes_train = np.argmax(y_pred_train, axis=1)

y_true_classes_test = np.argmax(y_test, axis=1)
y_true_classes_train = np.argmax(y_train, axis=1)

test_report = classification_report(y_true_classes_test, y_pred_classes_test, target_names=cm_labels)
train_report = classification_report(y_true_classes_train, y_pred_classes_train, target_names=cm_labels)

test_cm = confusion_matrix(y_true_classes_test, y_pred_classes_test)
train_cm = confusion_matrix(y_true_classes_train, y_pred_classes_train)

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

print(f"Train Set Classification report:\n {train_report}\n")
plot_confusion_matrix(train_cm, cm_labels, 'Train Set Confusion Matrix')

print(f"Test Set Classification report:\n {test_report}\n")
plot_confusion_matrix(test_cm, cm_labels, 'Test Set Confusion Matrix')

# Calculate additional metrics
def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    jaccard = jaccard_score(y_true_classes_test, y_pred_classes_test)
    return accuracy, sensitivity, specificity, precision, jaccard

train_accuracy, train_sensitivity, train_specificity, train_precision, train_jaccard = calculate_metrics(train_cm)
test_accuracy, test_sensitivity, test_specificity, test_precision, test_jaccard = calculate_metrics(test_cm)

print(f"Train Set Metrics:\n Accuracy: {train_accuracy:.4f}\n Sensitivity: {train_sensitivity:.4f}\n Specificity: {train_specificity:.4f}\n Precision: {train_precision:.4f}\n Jaccard Index: {train_jaccard:.4f}\n")
print(f"Test Set Metrics:\n Accuracy: {test_accuracy:.4f}\n Sensitivity: {test_sensitivity:.4f}\n Specificity: {test_specificity:.4f}\n Precision: {test_precision:.4f}\n Jaccard Index: {test_jaccard:.4f}\n")

# Plot training history
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

val_acc_values = history_dict['val_accuracy']
acc = history_dict['accuracy']

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
model_save_path = "F:/Dataset/archive/cnn_model.keras"
model.save(model_save_path)
