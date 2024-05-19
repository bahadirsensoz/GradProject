# GradProject

# Mammography Image Classification Manual

This guide provides instructions for setting up and using the mammography image classification script. The script is designed to process mammography images using various classifiers and evaluate their performance with multiple metrics.

## System Requirements / Prerequisites

Before running the script, please ensure you have the following setup:

+ Python 3.6 or later
+ pip (Python package installer)
+ VRAM >= 8GB, 24GB recommended
+ RAM > 16GB, 32GB+ recommended
+ Disk > 50GB
+ >500GB for large datasets

## Required Python Libraries

### For different scripts:

- **naivebayes.py**
  + `pip install psutil scikit-learn glob2 pillow numpy torchvision`
- **naivebayeswithbaggingboosting.py**
  + `pip install psutil scikit-learn pillow numpy torchvision glob2`
- **deneme2.py**
  + `pip install torch torchvision scikit-learn pillow glob2`
- **randomforestvit.py**
  + `pip install torch torchvision scikit-learn pillow glob2 numpy`
- **majorityvoting.py**
  + `pip install numpy os-sys glob2 pillow scikit-learn torch torchvision`
- **duzgundeneme.py**
  + `pip install matplotlib seaborn numpy pandas tensorflow opencv-python scikit-learn`
- **cuda_ver.py**
  + `pip install torch torchvision scikit-learn pillow glob2 psutil`

## Important Settings

For all instances of classifiers, ensure the dataset is correctly set up, for instance:

```python
dataset = MammographyDataset(data_dir="C:/Users/canok/OneDrive/Masaüstü/bitirme/jpeg", data_transform=transform)
```
Explanation: First, we are defining the location of dataset, and following that, we are specifying the transformations to be applied to each image in that dataset by using the next parameter in that function.

## Script Details in Basic Form

The script performs the following steps (example for naïve bayes classifier):

1. **Imports necessary libraries**.
2. **Defines transformations for image preprocessing**.
3. **Creates a `MammographyDataset` class for loading and transforming images**.
4. **Initializes the dataset and transforms images**.
5. **Monitors runtime and resource usage** (plus additional metrics).
6. **Transforms images and splits the data**.
7. **Trains a Naive Bayes classifier**.
8. **Evaluates the model's performance** using metrics like accuracy, F1 score, recall, and harmonic mean.
9. **Performs cross-validation**.
10. **Ends resource monitoring and prints results**.

## Example Usage

After running the script, you will see output similar to the following (example for naive bayes classifier):

```plaintext
Calculations started...
Accuracy on test set (Naive Bayes): 82.93%
F1 Score result: 82.76%
Recall result: 82.64%
Harmonic Mean of Accuracy, F1 Score, and Recall: 82.78%
Cross-Validation F1 Score: [0.80486515, 0.67876515, 0.79669421, 0.77795753, 0.83602151]
Mean F1 Accuracy: 0.7788607109671192
Runtime of the program is 18.37 seconds.
Initial CPU usage: 0.4%, Final CPU usage: 1.0%
Initial Memory usage: 33.5%, Final Memory usage: 34.3%

