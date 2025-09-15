import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # save images and labels of each directory
    images = []
    labels = []
    
    for subdirectory in range(NUM_CATEGORIES):
        subdirname = str(subdirectory)
        subdirectory_path = os.path.join(data_dir, subdirname)

        for filename in os.listdir(subdirectory_path):   # os.listdir returns a list containing the names of all files in the directory
            file_path = os.path.join(data_dir, subdirname, filename)
            img = cv2.imread(file_path)   # cv2.imread returns the image as a nparray

            # resize the numpy array to correct width and height
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # save to images and labels list
            images.append(img_resized)
            labels.append(subdirname)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # create a neural network
    model = tf.keras.models.Sequential()

    # define input shape
    model.add(tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # apply image convolution(extracts useful features) and pooling(reduces size of feature maps)
    # add 4 convolutional layers with 32 filters and 3x3 kernel matrix, using ReLU activation function
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))  
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))  
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))  
    # add a pooling layer with 2x2 pool size and stride of 2
    model.add(tf.keras.layers.MaxPooling2D((2, 2), 2))
    # apply dropout with a rate of 10% on the pooling layer
    model.add(tf.keras.layers.Dropout(0.1))

    # flatten 3D output of images to 1D to match output layer dimensions
    model.add(tf.keras.layers.Flatten())

    # pass into hidden layers
    # add a densely connected hidden layer with 256 nodes, using ReLU activation function
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    # add an output layer with the same number of nodes as the number of categories, using softmax activation function
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    
    model.summary()

    # compile neural network
    model.compile(
        optimizer="RMSprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
