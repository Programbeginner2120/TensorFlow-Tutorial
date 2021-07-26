import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


""" Loading in and preparing mnist dataset, then converting samples from integers to floating point numbers """
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

""" Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training: """
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 28 x 28 for 784 pixel drawn number
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

""" For each example the model returns a vector of "logits" or "log-odds" scores, one for each class. """
predictions = model(x_train[:1]).numpy()
# output for predictions: [[ 0.06980533 -0.09669013 -0.02087142  0.04041972 -0.0512895   0.37013057
#   -0.1566553   0.22757633 -0.3105838   0.37073052]]

""" The tf.nn.softmax function converts these logits to "probabilities" for each class: """
tf.nn.softmax(predictions).numpy()

# res_tensor = tf.nn.softmax(predictions).numpy()
# print(res_tensor)
# Output for this is [[0.08572375 0.09730355 0.15203974 0.11487947 0.0761406  0.12175011
#   0.11011975 0.1158217  0.0665093  0.05971201]]

"""The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for
each example."""
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This loss is equal to the negative log probability of the true class: It is zero if the model is sure of the correct
# class. This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be
# close to -tf.math.log(1/10) ~= 2.3."""

# print(loss_fn(y_train[:1], predictions).numpy())  # --> This will be ~= 2.3

"""Configures the model for training."""
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

"""The Model.fit method adjusts the model parameters to minimize the loss:"""
model.fit(x_train, y_train, epochs=5)  # Only training this model for 5 iterations, or "epochs"

"""The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set"."""
model.evaluate(x_test, y_test, verbose=2)  # now that we've trained the model using training data, we test the model
                                           # using the testing data

"""The image classifier is now trained to ~98% accuracy on this dataset. f you want your model to return a probability, 
you can wrap the trained model, and attach the softmax to it:"""
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# displays a tensor of shape (5,10) with each pointwise value in each array representing how likely the model believes
# the data to be the corresponding digit e.g. arr[0] probability for 0, arr[1] for 1, etc.
print(probability_model(x_test[:5]))






















