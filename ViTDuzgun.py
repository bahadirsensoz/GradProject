import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from transformers import TFViTModel, ViTImageProcessor
from tensorflow_addons.optimizers import AdamW
import tensorflow_addons as tfa
import matplotlib.image as mpimg
import pickle
from imblearn.over_sampling import SMOTE

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load metadata
meta_data_path = 'F:/Dataset/archive/csv/meta.csv'
dicom_info_path = 'F:/Dataset/archive/csv/dicom_info.csv'
mass_train_path = 'F:/Dataset/archive/csv/mass_case_description_train_set.csv'
mass_test_path = 'F:/Dataset/archive/csv/mass_case_description_test_set.csv'
jpeg_dir = 'F:/Dataset/archive/jpeg'

df_meta = pd.read_csv(meta_data_path)
df_dicom = pd.read_csv(dicom_info_path, delimiter=';')

# Extract unique series descriptions
series_descriptions = df_dicom.SeriesDescription.unique()

# Filter paths based on series descriptions
cropped_image_paths = df_dicom[df_dicom.SeriesDescription == 'cropped images'].image_path
full_mammo_paths = df_dicom[df_dicom.SeriesDescription == 'full mammogram images'].image_path
roi_image_paths = df_dicom[df_dicom.SeriesDescription == 'ROI mask images'].image_path

# Update image paths
cropped_image_paths = cropped_image_paths.str.replace('CBIS-DDSM/jpeg', jpeg_dir)
full_mammo_paths = full_mammo_paths.str.replace('CBIS-DDSM/jpeg', jpeg_dir)
roi_image_paths = roi_image_paths.str.replace('CBIS-DDSM/jpeg', jpeg_dir)

# Organize image paths in dictionaries
full_mammo_dict = {path.split("/")[4]: path for path in full_mammo_paths}
cropped_images_dict = {path.split("/")[4]: path for path in cropped_image_paths}
roi_img_dict = {path.split("/")[4]: path for path in roi_image_paths}

# Load mass case description datasets
mass_train = pd.read_csv(mass_train_path)
mass_test = pd.read_csv(mass_test_path)

# Rename columns for consistency
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

# Fix image paths in datasets
def update_image_paths(df):
    for idx, row in df.iterrows():
        full_image_key = row['image_file_path'].split("/")[2]
        cropped_image_key = row['cropped_image_file_path'].split("/")[2] if pd.notna(row['cropped_image_file_path']) else None

        if full_image_key in full_mammo_dict:
            df.at[idx, 'image_file_path'] = full_mammo_dict[full_image_key]
        if cropped_image_key and cropped_image_key in cropped_images_dict:
            df.at[idx, 'cropped_image_file_path'] = cropped_images_dict[cropped_image_key]

update_image_paths(mass_train)
update_image_paths(mass_test)

# Handle missing values
mass_train['mass_shape'].bfill(inplace=True)
mass_train['mass_margins'].bfill(inplace=True)
mass_test['mass_margins'].bfill(inplace=True)

# Function to resize and display images
def display_images(column, number):
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    for index, row in mass_train.head(number_to_visualize).iterrows():
        image_path = row[column]
        if image_path and os.path.exists(image_path):
            image = mpimg.imread(image_path)
            if len(image.shape) == 2:  # if grayscale, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            ax = axes[index]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"{row['pathology']}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()

display_images('image_file_path', 5)
display_images('cropped_image_file_path', 5)

# Merge training and test datasets
full_mass_data = pd.concat([mass_train, mass_test], axis=0)

# Resize images to a manageable size
def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)

# Extract processed images and labels in batches
def load_and_preprocess_images(image_paths, feature_extractor, batch_size=100):
    images = []
    valid_paths = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            if path and os.path.exists(path):
                image = mpimg.imread(os.path.abspath(path))
                if len(image.shape) == 2:  # if grayscale, convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = resize_image(image)
                batch_images.append(image)
                valid_paths.append(path)
        if batch_images:
            inputs = feature_extractor(images=batch_images, return_tensors="tf")
            images.append(inputs['pixel_values'])
    if not images:
        raise ValueError("No valid images found for preprocessing.")
    return tf.concat(images, axis=0), valid_paths

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
X, valid_image_paths = load_and_preprocess_images(full_mass_data['image_file_path'], feature_extractor)

# Ensure valid paths are correctly assigned back to the DataFrame for further processing
full_mass_data = full_mass_data[full_mass_data['image_file_path'].isin(valid_image_paths)]

y = full_mass_data['pathology'].replace({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}).values

# Convert to TensorFlow tensors
X = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(y, dtype=tf.int32)

# Split data into train, test, and validation sets (70, 20, 10)
X_train, X_temp, y_train, y_temp = train_test_split(X.numpy(), y.numpy(), test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Convert back to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
X_val = tf.convert_to_tensor(X_val)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)
y_val = tf.convert_to_tensor(y_val)

train_class_counts = np.bincount(y_train.numpy())
test_class_counts = np.bincount(y_test.numpy())

print("Number of occurrences of each class in y_train:")
print(f"Class 0: {train_class_counts[0]}")
print(f"Class 1: {train_class_counts[1]}")

print("\nNumber of occurrences of each class in y_test:")
print(f"Class 0: {test_class_counts[0]}")
print(f"Class 1: {test_class_counts[1]}")

# Convert integer labels to one-hot encoded labels
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
y_val = to_categorical(y_val, 2)

# Augment data
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Apply augmentation to training data
train_data_augmented = train_datagen.flow(X_train.numpy(), y_train, batch_size=32)


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
    vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = tf.keras.Input(shape=(3, 224, 224), name='pixel_values', dtype=tf.float32)
    vit_outputs = vit_model(pixel_values=inputs).last_hidden_state
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(vit_outputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(pooled_output)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Resample the training data using SMOTE to address class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.numpy().reshape(X_train.shape[0], -1), y_train.argmax(axis=1))
X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], 3, 224, 224)
y_train_resampled = to_categorical(y_train_resampled, 2)

# Convert back to tensors
X_train_resampled = tf.convert_to_tensor(X_train_resampled)
y_train_resampled = tf.convert_to_tensor(y_train_resampled)

train_data_augmented = train_datagen.flow(X_train_resampled.numpy(), y_train_resampled, batch_size=32)

# Train model
history = model.fit(
    train_data_augmented,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, lr_scheduler]
)

model.summary()

model.evaluate(X_test, y_test)

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

model_save_path = "F:/Dataset/archive/vit_batch32_earlyStop_CategoricalCrossEntropy.keras"
model.save(model_save_path)
