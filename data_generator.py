import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset from TensorFlow datasets (automatically downloads if not present)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape the data for ImageDataGenerator (add a single channel for grayscale images)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Augment the data using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=22,        # Rotate images by 15 degrees
    width_shift_range=0.1,    # Shift images horizontally by 10%
    height_shift_range=0.1,   # Shift images vertically by 10%
    shear_range=0.1,          # Shearing transformation
    zoom_range=0.1,           # Random zooming
    fill_mode='nearest'       # Fill empty pixels after rotation/shift
)

# Fit the ImageDataGenerator on the training data
datagen.fit(x_train)

# Example function to display a limited number of original and augmented images
def show_augmented_images(datagen, x_train, y_train, num_images=10):
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        sample_image = x_train[i]
        sample_label = y_train[i]

        # Display the original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(sample_image.squeeze(), cmap='gray')
        plt.title(f'Original {sample_label}')
        plt.axis('off')

        # Generate one augmented version of the image and display it
        for batch in datagen.flow(np.expand_dims(sample_image, axis=0), batch_size=1):
            augmented_image = batch.squeeze()
            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(augmented_image, cmap='gray')
            plt.title(f'Augmented {sample_label}')
            plt.axis('off')
            break

    plt.tight_layout()
    plt.show()

# Display 10 original and augmented images
show_augmented_images(datagen, x_train, y_train, num_images=10)
