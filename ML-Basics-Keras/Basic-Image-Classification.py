import tensorflow as tf  # contains Tensorflow and keras API
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
train_images = train_images / 255.0
test_images = test_images / 255.0

"""Test to verify correctness of format of data"""
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])  # --> Removes ticks on the x-axis i.e. plain graph for figure
#     plt.yticks([])  # --> Removes ticks on the y-axis
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

"""Building the model"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flattening the 28 x 28 image to 1D array of 784 elements
    tf.keras.layers.Dense(128, activation='relu'),  # this layer is fully connected and has 128 nodes
    tf.keras.layers.Dense(10)  # fully connected output layer with 10 nodes, returns logits array with length 10
])

"""Compiling model using built in optimizer, loss function and measuring any metrics desired during training"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""fitting the model to the training data, commences training"""
model.fit(train_images, train_labels, epochs=10)

"""Evaluating model accuracy on training examples using model.evaluate, then displaying"""
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

"""Able to make predictions about testing images by creating a probability model and attach a softmax
layer to convert the logits to probabilities"""
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)  # returns tensor of predictions for each inputted test image
# print(predictions[0])  # Displays probability array
mapping_predicted = np.argmax(predictions[0])  # returns mapping number of predicted article of clothing
# print(mapping_predicted)
# print(mapping_predicted == test_labels[0])  # prints if it is correct or not

"""plots image of given testing example"""
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel("{} {:2.0f}% ({}))".format(class_names[predicted_label],
                                          100 * np.max(predictions_array),
                                          class_names[true_label],
                                          color=color))

"""Plots bar graph representing value array for test image"""
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


"""Plotting for first test image"""
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

"""Plotting for thirteenth test image"""
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# TODO: FINISH THIS PART OF TUTORIAL
















