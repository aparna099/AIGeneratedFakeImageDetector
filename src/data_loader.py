import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_images_from_directory(directory):
    """
    Loads images and corresponding labels from the specified directory.

    Args:
        directory (str): Path to the directory containing the image subdirectories.

    Returns:
        tuple: Arrays of loaded images and labels.
    """
    print('Loading images...')

    images = []
    labels = []

    # Iterate over the subdirectories in the directory
    for subdirectory in os.listdir(directory):
        subdir_path = os.path.join(directory, subdirectory)
        if os.path.isdir(subdir_path):
            # Assign label based on the subdirectory name
            label = 1 if subdirectory == 'Real' else 0

            # Load images from the fake or real subfolder
            for filename in os.listdir(subdir_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    # Load the image using PIL
                    img_path = os.path.join(subdir_path, filename)
                    img = Image.open(img_path)

                    # Append the image and its label to the lists
                    images.append(img)
                    labels.append(label)
    
    # Preprocess the images
    images_processed = preprocess_images(images)  

    # Preprocess the labels
    labels_processed = preprocess_labels(labels)

    return np.array(images_processed), np.array(labels_processed)

def preprocess_images(images):
    """
    Preprocesses the loaded images.

    Args:
        images (numpy.ndarray): Array of images.

    Returns:
        numpy.ndarray: Preprocessed images.
    """
    print('Data preprocessing...')

    processed_images = []

    for img in images:
        # Preprocess the image (resize, convert to grayscale, etc.)
        img = img.resize((32, 32))  # Resize the image to desired dimensions
        img = img.convert('RGB')  # Convert the image to RGB if needed

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Normalize the pixel values between 0 and 1
        img_array = img_array / 255.0

        # Append the processed image to the list
        processed_images.append(img_array)

    return np.array(processed_images)

def preprocess_labels(labels):
    """
    Preprocesses the labels.

    Args:
        labels (numpy.ndarray): Array of labels.

    Returns:
        numpy.ndarray: Preprocessed labels.
    """
    # Perform one-hot encoding on the labels
    encoded_labels = tf.keras.utils.to_categorical(labels)

    return encoded_labels
def split_data(images, labels):
    """
    Splits the data into training and test sets.

    Args:
        images (numpy.ndarray): Array of images.
        labels (numpy.ndarray): Array of labels.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    print('Splitting data into train and test set...')
    
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

    return x_train, x_test, y_train, y_test
