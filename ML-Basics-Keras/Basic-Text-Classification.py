import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print(tf.__version__)

"""Downloading and extracting the IMDB dataset"""
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
# print(os.listdir(dataset_dir))

"""accessing subdirectory with dataset_dir, i.e. train_dir"""
train_dir = os.path.join(dataset_dir, 'train')
# print(os.listdir(train_dir))

"""Taking a look at the movie review text files within the aclImdb/train/pos directory"""
# sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
# with open(sample_file) as f:
#     print(f.read())  # NOTE: THE WITH KEYWORD IS USED IN PYTHON EXCEPTION HANDLING TO MAKE CODE CLEANER!!!

"""Removing the one extra directory within train_dir, leaving only pos and neg directories for binary classification"""
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

"""Creating a validation with an 80:20 split of the training data; already have training and testing datasets"""
batch_size = 42
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

"""you can train a model by passing a dataset directly to model.fit. If you're new to tf.data, you can also iterate over
 the dataset and print out a few examples as follows:"""
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("label 1 corresponds to", raw_train_ds.class_names[1])

#TODO: FINISH UP THIS PART OF TUTORIAL NEXT TIME, START WITH BLUE STAR AND CREATING VALIDATION AND TEST SETS









