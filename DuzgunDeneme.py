import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import os


df_meta = pd.read_csv('F:/Dataset/archive/csv/meta.csv')
print(df_meta.head())



df_dicom = pd.read_csv('F:/Dataset/archive/csv/dicom_info.csv')
print(df_dicom.head())


print(df_dicom.SeriesDescription.unique())

cropped_images = df_dicom[df_dicom.SeriesDescription=='cropped images'].image_path

print(cropped_images.head())

full_mammo = df_dicom[df_dicom.SeriesDescription=='full mammogram images'].image_path

print(full_mammo.head())

roi_img = df_dicom[df_dicom.SeriesDescription=='ROI mask images'].image_path
print(roi_img.head(5))

imdir = 'F:/Dataset/archive/jpeg'

cropped_images = cropped_images.replace('CBIS-DDSM/jpeg', imdir, regex=True)
full_mammo = full_mammo.replace('CBIS-DDSM/jpeg', imdir, regex=True)
roi_img = roi_img.replace('CBIS-DDSM/jpeg', imdir, regex=True)

# view new paths
print('Cropped Images paths:\n')
print(cropped_images.iloc[0])
print('Full mammo Images paths:\n')
print(full_mammo.iloc[0])
print('ROI Mask Images paths:\n')
print(roi_img.iloc[0])

# organize image paths
full_mammo_dict = dict()
cropped_images_dict = dict()
roi_img_dict = dict()

for dicom in full_mammo:
    key = dicom.split("/")[4]
    full_mammo_dict[key] = dicom
for dicom in cropped_images:
    key = dicom.split("/")[4]
    cropped_images_dict[key] = dicom
for dicom in roi_img:
    key = dicom.split("/")[4]
    roi_img[key] = dicom

# view keys
print(next(iter((full_mammo_dict.items()))))



# load the mass dataset
mass_train = pd.read_csv('F:/Dataset/archive/csv/mass_case_description_train_set.csv')
mass_test = pd.read_csv('F:/Dataset/archive/csv/mass_case_description_test_set.csv')

print(mass_train.head())
print(mass_test.head())


def fix_image_path(data):
    """correct dicom paths to correct image paths"""
    for index, img in enumerate(data.values):
        img_name = img[11].split("/")[2]
        data.iloc[index, 11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        data.iloc[index, 12] = cropped_images_dict[img_name]


# apply to datasets
fix_image_path(mass_train)
fix_image_path(mass_test)

print(mass_train.pathology.unique())

print(mass_train.info())

mass_train = mass_train.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

mass_test = mass_test.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})


print(mass_train.info())

print(mass_train.isnull().sum())
print(mass_train.describe())


mass_train['mass_shape'] = mass_train['mass_shape'].fillna(method='bfill')
mass_train['mass_margins'] = mass_train['mass_margins'].fillna(method='bfill')

print(mass_train.isnull().sum())
print(f'Shape of mass_train: {mass_train.shape}')
print(f'Shape of mass_test: {mass_test.shape}')

print(mass_test.isnull().sum())

mass_test['mass_margins'] = mass_test['mass_margins'].fillna(method='bfill')

print(mass_test.isnull().sum())

value = mass_train['pathology'].value_counts()
plt.figure(figsize=(8,6))
plt.pie(value, labels=value.index, autopct='%1.1f%%')
plt.title('Breast Cancer Mass Types', fontsize=14)
plt.savefig('F:/Dataset/archivearchivepathology_distributions_red.png')
plt.show()


plt.figure(figsize=(8,6))
sns.countplot(mass_train, y='assessment', hue='pathology', palette='viridis')
plt.title('Breast Cancer Assessment\n\n 0: Undetermined || 1: Well Differentiated\n2: Moderately differentiated || 3: Poorly DIfferentiated\n4-5: Undifferentiated',
          fontsize=12)
plt.ylabel('Assessment Grade')
plt.xlabel('Count')
plt.savefig('F:/Dataset/archive/breast_assessment_red.png')
plt.show()


plt.figure(figsize=(8,6))
sns.countplot(mass_train, x='subtlety', palette='viridis')
plt.title('Breast Cancer Mass Subtlety', fontsize=12)
plt.xlabel('Subtlety Grade')
plt.ylabel('Count')
plt.savefig('F:/Dataset/archive/cancer_subtlety_red.png')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(mass_train, x='mass_shape', hue='pathology')
plt.title('Mass Shape Distribution by Pathology', fontsize=14)
plt.xlabel('Mass Shape')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Pathology Count')
plt.legend()
plt.savefig('F:/Dataset/archive/mass_pathology_red.png')
plt.show()


plt.figure(figsize=(8,6))

sns.countplot(mass_train, x='breast_density', hue='pathology')
plt.title('Breast Density vs Pathology\n\n1: fatty || 2: Scattered Fibroglandular Density\n3: Heterogenously Dense || 4: Extremely Dense',
          fontsize=14)
plt.xlabel('Density Grades')
plt.ylabel('Count')
plt.legend()
plt.savefig('F:/Dataset/archive/density_pathology_red.png')
plt.show()


def display_images(column, number):
    """displays images in dataset"""
    # create figure and axes
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))

    # Loop through rows and display images
    for index, row in mass_train.head(number_to_visualize).iterrows():
        image_path = row[column]
        image = mpimg.imread(image_path)
        ax = axes[index]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"{row['pathology']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


print('Full Mammograms:\n')
display_images('image_file_path', 10)
print('Cropped Mammograms:\n')
display_images('cropped_image_file_path', 10)

def image_processor(image_path, target_size):
    """Preprocess images for CNN model"""
    absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(absolute_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image_array = image / 255.0
    return image_array

# Merge datasets
full_mass = pd.concat([mass_train, mass_test], axis=0)