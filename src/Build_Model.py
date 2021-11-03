## ############################################################################
## Imports
## ############################################################################
import functions as f
import pickle
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

## ############################################################################
## Initialization
## ############################################################################
now = datetime.now()
strNow = now.strftime("%Y-%m-%d-%H-%M-%S")
image_directory = './image_data'
labels_filepath = './class_labels/labels_map_' + strNow + '.pkl'
model_filepath = './model/best_model_' + strNow + '.hdf5'


## ############################################################################
## Load and Preprocess Images
## ############################################################################
images, labels, paths = f.getImagesAndClasses(image_directory)
images_list, labels_list = f.expand_with_rotated_images(np.array(images), labels, 12)
images = np.array([f.resize_image(np.array(x), 224) for x in images_list])
images = np.array([f.normalize_image(x) for x in images])


## ############################################################################
## One-Hot Encode
## ############################################################################
## For each image, we know it's class name.  That's listed in labels.
## We need to create an integer for each unique label.
integer_encoding_map = f.get_integer_map(labels_list)

## Then, create a new list that is numeric labels
numeric_labels = []
for i in labels_list:
    numeric_labels.append(integer_encoding_map.index(i))

## Then, we need to create one last list which is converts each numeric 
## label into a one-hot representation
one_hot_labels = []
for i in numeric_labels:
    one_hot_labels.append(f.convert_to_one_hot(i, integer_encoding_map))

## Save the integer_encoding_map to use later when decoding predictions
with open(labels_filepath, 'wb') as pickle_file:
    pickle.dump(integer_encoding_map, pickle_file)


## ############################################################################
## Split the images (and labels) into training and validation sets
## ############################################################################
train_images, val_images, train_labels, val_labels = train_test_split(images, np.array(one_hot_labels), test_size=0.20, random_state=42)


## ############################################################################
## Create a neural network model
## ############################################################################
#model = tf.keras.Sequential([
#    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
#    tf.keras.layers.Dense(128, activation='relu'),
#    tf.keras.layers.Dense(len(set(labels_list)))
#])

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(set(labels_list))))

## ############################################################################
## Compile the model
## ############################################################################
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['acc'])


## ############################################################################
## Define callbacks
## ############################################################################
es = keras.callbacks.EarlyStopping(monitor='val_acc', # loss
                                   mode='max', # min
                                   verbose=1, 
                                   patience=120, 
                                   min_delta=0.0001)
mc = keras.callbacks.ModelCheckpoint(model_filepath, 
                                     monitor='val_acc', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max')


## ############################################################################
## Train the model
## ############################################################################
history = model.fit(train_images, 
                    train_labels, 
                    callbacks = [es, mc],
                    epochs    = 600,
                    validation_data=(val_images, val_labels))


## ############################################################################
## View the training statistics
## ############################################################################
best_idx = np.argmax(history.history['val_acc'])
best_acc = history.history['val_acc'][best_idx]

print('')
print('----------------------------------------------')
print('MODEL TRAINING REPORT')
print('----------------------------------------------')
print('Best Validation Accuracy Achieved: {}'.format(best_acc))
print('Best Model saved at              : {}'.format(model_filepath))
print('Class label map saved at         : {}'.format(labels_filepath))
print('')