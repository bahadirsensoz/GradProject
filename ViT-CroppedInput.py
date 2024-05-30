import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, jaccard_score
import seaborn as sns
from transformers import TFViTModel, ViTImageProcessor
from tensorflow_addons.optimizers import AdamW
import tensorflow_addons as tfa
import matplotlib.image as mpimg
from imblearn.over_sampling import SMOTE

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory limit to a higher value (e.g., 8GB)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])
    except RuntimeError as e:
        print(e)

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
def update_image_paths_with_crops(df):
    for idx, row in df.iterrows():
        full_image_key = row['image_file_path'].split("/")[2]
        cropped_image_key = row['cropped_image_file_path'].split("/")[2] if pd.notna(
            row['cropped_image_file_path']) else None

        if full_image_key in full_mammo_dict:
            df.at[idx, 'image_file_path'] = full_mammo_dict[full_image_key]
        if cropped_image_key and cropped_image_key in cropped_images_dict:
            df.at[idx, 'cropped_image_file_path'] = cropped_images_dict[cropped_image_key]


update_image_paths_with_crops(mass_train)
update_image_paths_with_crops(mass_test)

# Handle missing values
mass_train['mass_shape'].bfill(inplace=True)
mass_train['mass_margins'].bfill(inplace=True)
mass_test['mass_margins'].bfill(inplace=True)


# Function to resize and display images
def display_images(column1, column2, number):
    number_to_visualize = number
    rows = 2
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    for index, row in mass_train.head(number_to_visualize).iterrows():
        full_image_path = row[column1]
        cropped_image_path = row[column2]
        if full_image_path and os.path.exists(full_image_path):
            full_image = mpimg.imread(full_image_path)
            if len(full_image.shape) == 2:  # if grayscale, convert to RGB
                full_image = cv2.cvtColor(full_image, cv2.COLOR_GRAY2RGB)
            ax = axes[0, index]
            ax.imshow(full_image, cmap='gray')
            ax.set_title(f"Full: {row['pathology']}")
            ax.axis('off')
        if cropped_image_path and os.path.exists(cropped_image_path):
            cropped_image = mpimg.imread(cropped_image_path)
            if len(cropped_image.shape) == 2:  # if grayscale, convert to RGB
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
            ax = axes[1, index]
            ax.imshow(cropped_image, cmap='gray')
            ax.set_title(f"Cropped: {row['pathology']}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()


display_images('image_file_path', 'cropped_image_file_path', 5)

# Merge training and test datasets
full_mass_data = pd.concat([mass_train, mass_test], axis=0)


# Resize images to a manageable size
def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)


# Extract processed images and labels in batches
def load_and_preprocess_images_with_crops(df, feature_extractor, batch_size=100):
    full_images = []
    cropped_images = []
    valid_paths = []

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_full_images = []
        batch_cropped_images = []

        for idx, row in batch_df.iterrows():
            full_image_path = row['image_file_path']
            cropped_image_path = row['cropped_image_file_path']
            if full_image_path and os.path.exists(full_image_path):
                full_image = mpimg.imread(full_image_path)
                if len(full_image.shape) == 2:  # if grayscale, convert to RGB
                    full_image = cv2.cvtColor(full_image, cv2.COLOR_GRAY2RGB)
                full_image = resize_image(full_image)
                batch_full_images.append(full_image)
                valid_paths.append(full_image_path)
            if cropped_image_path and os.path.exists(cropped_image_path):
                cropped_image = mpimg.imread(cropped_image_path)
                if len(cropped_image.shape) == 2:  # if grayscale, convert to RGB
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
                cropped_image = resize_image(cropped_image)
                batch_cropped_images.append(cropped_image)

        if batch_full_images:
            full_inputs = feature_extractor(images=batch_full_images, return_tensors="tf")
            full_images.append(full_inputs['pixel_values'])

        if batch_cropped_images:
            cropped_inputs = feature_extractor(images=batch_cropped_images, return_tensors="tf")
            cropped_images.append(cropped_inputs['pixel_values'])

    if not full_images or not cropped_images:
        raise ValueError("No valid images found for preprocessing.")

    return tf.concat(full_images, axis=0), tf.concat(cropped_images, axis=0), valid_paths


feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
X_full, X_cropped, valid_image_paths = load_and_preprocess_images_with_crops(full_mass_data, feature_extractor)
full_mass_data = full_mass_data[full_mass_data['image_file_path'].isin(valid_image_paths)]
y = full_mass_data['pathology'].replace({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}).values

# Convert to TensorFlow tensors
X_full = tf.convert_to_tensor(X_full)
X_cropped = tf.convert_to_tensor(X_cropped)
y = tf.convert_to_tensor(y, dtype=tf.int32)

# Split data into train, test, and validation sets (70, 20, 10)
X_full_train, X_full_temp, y_train, y_temp = train_test_split(X_full.numpy(), y.numpy(), test_size=0.3, random_state=42)
X_cropped_train, X_cropped_temp = train_test_split(X_cropped.numpy(), test_size=0.3, random_state=42)

X_full_test, X_full_val, y_test, y_val = train_test_split(X_full_temp, y_temp, test_size=0.33, random_state=42)
X_cropped_test, X_cropped_val = train_test_split(X_cropped_temp, test_size=0.33, random_state=42)

# Convert back to TensorFlow tensors
X_full_train = tf.convert_to_tensor(X_full_train)
X_cropped_train = tf.convert_to_tensor(X_cropped_train)
X_full_test = tf.convert_to_tensor(X_full_test)
X_cropped_test = tf.convert_to_tensor(X_cropped_test)
X_full_val = tf.convert_to_tensor(X_full_val)
X_cropped_val = tf.convert_to_tensor(X_cropped_val)

y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)
y_val = tf.convert_to_tensor(y_val)

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
train_data_augmented = train_datagen.flow([X_full_train.numpy(), X_cropped_train.numpy()], y_train, batch_size=25)

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define distributed training strategy
strategy = tf.distribute.MirroredStrategy()


# Define the model with two inputs
def create_vit_model_with_cropped_images():
    vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    full_input = tf.keras.Input(shape=(3, 224, 224), name='full_image', dtype=tf.float32)
    cropped_input = tf.keras.Input(shape=(3, 224, 224), name='cropped_image', dtype=tf.float32)

    full_vit_outputs = vit_model(pixel_values=full_input).last_hidden_state
    cropped_vit_outputs = vit_model(pixel_values=cropped_input).last_hidden_state

    full_pooled_output = tf.keras.layers.GlobalAveragePooling1D()(full_vit_outputs)
    cropped_pooled_output = tf.keras.layers.GlobalAveragePooling1D()(cropped_vit_outputs)

    concatenated = tf.keras.layers.Concatenate()([full_pooled_output, cropped_pooled_output])

    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(concatenated)

    model = tf.keras.Model(inputs=[full_input, cropped_input], outputs=outputs)

    model.compile(
        optimizer=AdamW(learning_rate=0.0001, weight_decay=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


with strategy.scope():
    model = create_vit_model_with_cropped_images()


# Define the learning rate schedule function
def lr_schedule(epoch, lr):
    if epoch < 7:
        return lr
    else:
        return lr * tf.math.exp(-0.01)


# Implement early stopping and learning rate scheduler
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Clear GPU memory before starting training
tf.keras.backend.clear_session()

# Train model
history = model.fit(
    train_data_augmented,
    epochs=100,
    validation_data=([X_full_val, X_cropped_val], y_val),
    callbacks=[early_stopping, lr_scheduler]
)

model.summary()

# Evaluate the model
model.evaluate([X_full_test, X_cropped_test], y_test)

# Plot confusion matrix and classification report
cm_labels = ['MALIGNANT', 'BENIGN']

y_pred_test = model.predict([X_full_test, X_cropped_test])
y_pred_train = model.predict([X_full_train, X_cropped_train])

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

# Calculate and print additional metrics
def calculate_additional_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    return sensitivity, specificity, accuracy, f1, jaccard

sensitivity, specificity, accuracy, f1, jaccard = calculate_additional_metrics(y_true_classes_test, y_pred_classes_test)

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Jaccard Index: {jaccard:.4f}")

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
