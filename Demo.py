import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from transformers import TFViTModel
import pydicom


# Register the custom layer
def get_custom_objects():
    return {"TFViTModel": TFViTModel}


# Load the trained model
model_path = "F:/Dataset/archive/0.79AccGoodSense.keras"
custom_objects = get_custom_objects()
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    full_image = cv2.resize(image, (224, 224))
    cropped_image = cv2.resize(image[50:274, 50:274], (224, 224))  # example crop

    # Convert to channels-first format
    full_image = np.transpose(full_image, (2, 0, 1))
    cropped_image = np.transpose(cropped_image, (2, 0, 1))

    full_image = np.expand_dims(full_image, axis=0)
    cropped_image = np.expand_dims(cropped_image, axis=0)

    return full_image, cropped_image


# Function to convert DICOM to JPEG
def dicom_to_jpeg(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    jpeg_path = dicom_path.replace('.dicom', '.jpg').replace('.dcm', '.jpg')
    cv2.imwrite(jpeg_path, image)
    return jpeg_path


# Function to convert PGM to JPEG
def pgm_to_jpeg(pgm_path):
    image = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    jpeg_path = pgm_path.replace('.pgm', '.jpg')
    cv2.imwrite(jpeg_path, image)
    return jpeg_path


# Function to predict image
def predict_image(image_path):
    full_image, cropped_image = preprocess_image(image_path)
    prediction = model.predict([full_image, cropped_image])
    class_index = np.argmax(prediction, axis=1)[0]
    class_name = "MALIGNANT" if class_index == 1 else "BENIGN"
    return class_name


# GUI Setup
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mammogram Image Classifier")
        self.label = tk.Label(root, text="Click to choose an Image to Classify", font=("Helvetica", 16))
        self.label.pack(pady=20)
        self.canvas = tk.Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack(pady=20)
        self.result_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=20)
        self.canvas.bind("<Button-1>", self.open_file)

    def open_file(self, event):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.dcm;*.dicom;*.pgm")])
        if file_path:
            if file_path.lower().endswith((".dcm", ".dicom")):
                jpeg_path = dicom_to_jpeg(file_path)
            elif file_path.lower().endswith(".pgm"):
                jpeg_path = pgm_to_jpeg(file_path)
            else:
                jpeg_path = file_path
            self.display_image(jpeg_path)
            result = predict_image(jpeg_path)
            self.result_label.config(text=f"Prediction: {result}")

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
