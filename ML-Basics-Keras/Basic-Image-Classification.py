import tensorflow as tf  # contains Tensorflow and keras API
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)  # prints version of Tensorflow installed on machine

"""Accessing and loading in data"""
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# NOTE: load_data() returns four numpy arrays, you can verify below
# print(type(train_images), type(train_labels), type(test_images), type(test_labels))

"""Python dict containing the mapping to articles of clothing"""
class_names = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
               7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# Outputting shape of train_images tensor, tensor of 60000 28 x 28 images, i.e. (60000, 28, 28) shape
# print(train_images.shape)
# print(len(train_images)  # --> length of the tensor is 60000
# print(train_labels)  # --> displays shortened array of 60000 digit mappings for the images

"""Using matplotlib to generate figure for first image and display pixel concentration using color density"""
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)  # When set to True, shows grid lines on figure
# plt.show()

"""Scaling train and test images to have values between 0 and 1, rather than between 0 and 255"""
train_images, test_images = train_images / 255.0, train_labels / 255.0

"""Test to verify correctness of format of data"""
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])  # --> Removes ticks on the x-axis i.e. plain graph for figure
    plt.yticks([])  # --> Removes ticks on the y-axis
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

"""Building the model"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flattening the 28 x 28 image to 1D array of 784 elements
    tf.keras.layers.Dense(128, activation='relu'),  # this layer is fully connected as has 128 nodes
    tf.keras.layers.Dense(10)  # fully connected output layer with 10 nodes, returns logits array with length 10
])

"""Compiling model using built in optimizer, loss function and measuring any metrics desired during training"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# TODO: FINISH UP THIS PART OF TUTORIAL TOMORROW














