import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf


def plot_loss_curves(history):
    """Returns separate loss curves for training and validation metrics.

    :param history: TensorFlow model History object.
    """

    loss = history.history['loss']
    accuracy = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    epochs = range(len(history.history['loss']))

    # Plot accuracy
    plt.plot(epochs, accuracy, color='green', label = "Training accuracy")
    plt.plot(epochs, val_acc, color = 'blue', label = "Validation accuracy")
    plt.title("Accuracies")
    plt.xlabel("Epochs")
    plt.legend()

    plt.figure()

    # Plot loss
    plt.plot(epochs, loss, color = 'red', label = "Training loss")
    plt.plot(epochs, val_loss, color = 'pink', label = "Validation loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show();

def load_image_tensorflow(image_file_path, image_scale = 224, scale = True):
    """Transform image from simple path to tensor format.

    :param image_file_path: The path to the image.
    :param image_scale: The scale you want your image to be (default 224).
    :param scale: Boolean value whether to apply scale or not.
    :return: image in tensor format (tensorflow.python.framework.ops.EagerTensor).
    """
    # Read image
    image = tf.io.read_file(image_file_path)
    # Decode tensor
    image = tf.io.decode_jpeg(image)
    # Resize the image
    image = tf.image.resize(image, [image_scale, image_scale])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return image/255.
    return image