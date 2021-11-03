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
from PIL import Image, ImageOps


## ############################################################################
## Initialization
## ############################################################################
model_filepath = './model/best_model_2021-11-02-17-06-04.hdf5'
labels_filepath = './class_labels/labels_map_2021-11-02-17-06-04.pkl'


## ############################################################################
## Load the model
## ############################################################################
model = keras.models.load_model(model_filepath)


## ############################################################################
## Load the labels
## ############################################################################
with open(labels_filepath, 'rb') as pickle_file:
    integer_encoding_map = pickle.load(pickle_file)


## ############################################################################
## List the image files
## ############################################################################
auricula_ringens_files = [
    './image_data/auricula_ringens/PAL-CL.01779e.png',
    './image_data/auricula_ringens/PAL-CL.01779d.png',
    './image_data/auricula_ringens/PAL-CL.01779c.png',
    './image_data/auricula_ringens/PAL-CL.01779b.png',
    './image_data/auricula_ringens/PAL-CL.01778i.png',
    './image_data/auricula_ringens/PAL-CL.01778c.png',
    './image_data/auricula_ringens/PAL-CL.01777e.png',
    './image_data/auricula_ringens/PAL-CL.01777d.png',
    './image_data/auricula_ringens/PAL-CL.01778b.png',
    './image_data/auricula_ringens/PAL-CL.01777b.png'
]


## ############################################################################
## Make a prediction for each image
## ############################################################################
predictions = f.make_predictions(auricula_ringens_files, model, integer_encoding_map)
print('')
print('AURICULA_RINGENS')
print('File, Prediction')
for i in range(len(auricula_ringens_files)):
    print('{},{}'.format(auricula_ringens_files[i], predictions[i]))
print('')


## ############################################################################
## Repeat with marginella files
## ############################################################################
marginella_files = [
    './image_data/marginella/PAL-CL.01745g.png',
    './image_data/marginella/PAL-CL.01745f.png',
    './image_data/marginella/PAL-CL.01745e.png',
    './image_data/marginella/PAL-CL.01744g.png',
    './image_data/marginella/PAL-CL.01744e.png',
    './image_data/marginella/PAL-CL.01743b.png',
    './image_data/marginella/PAL-CL.01741b.png',
    './image_data/marginella/PAL-CL.01740.png',
    './image_data/marginella/PAL-CL.01691g.png',
    './image_data/marginella/PAL-CL.01691f.png',
    './image_data/marginella/PAL-CL.01691e.png',
    './image_data/marginella/PAL-CL.01691d.png',
    './image_data/marginella/PAL-CL.01691c.png'
]
predictions = f.make_predictions(marginella_files, model, integer_encoding_map)
print('')
print('MARGINELLA')
print('File, Prediction')
for i in range(len(marginella_files)):
    print('{},{}'.format(marginella_files[i], predictions[i]))
print('')

## ############################################################################
## Repeat with erycina files
## ############################################################################
erycina_files = [
    './image_data/erycina/PAL-CL.00106j.png',
    './image_data/erycina/PAL-CL.00106i.png',
    './image_data/erycina/PAL-CL.00106h.png',
    './image_data/erycina/PAL-CL.00106g.png',
    './image_data/erycina/PAL-CL.00106f.png',
    './image_data/erycina/PAL-CL.00106e.png',
    './image_data/erycina/PAL-CL.00105j.png',
    './image_data/erycina/PAL-CL.00105i.png',
    './image_data/erycina/PAL-CL.00105h.png',
    './image_data/erycina/PAL-CL.00105f.png',
    './image_data/erycina/PAL-CL.00105e.png',
    './image_data/erycina/PAL-CL.00105b.png'
]
predictions = f.make_predictions(erycina_files, model, integer_encoding_map)
print('')
print('ERYCINA')
print('File, Prediction')
for i in range(len(erycina_files)):
    print('{},{}'.format(erycina_files[i], predictions[i]))
print('')

## ############################################################################
## Repeat with trochus files
## ############################################################################
trochus_files = [
    './image_data/trochus/PAL-CL.00921b.png',
    './image_data/trochus/PAL-CL.00920.png',
    './image_data/trochus/PAL-CL.00919c.png',
    './image_data/trochus/PAL-CL.00903.png',
    './image_data/trochus/PAL-CL.00901f.png',
    './image_data/trochus/PAL-CL.00901e.png',
    './image_data/trochus/PAL-CL.00901d.png',
    './image_data/trochus/PAL-CL.00901c.png',
    './image_data/trochus/PAL-CL.00895.png',
    './image_data/trochus/PAL-CL.00891.png',
    './image_data/trochus/PAL-CL.00890.png',
    './image_data/trochus/PAL-CL.00884.png'
]
predictions = f.make_predictions(trochus_files, model, integer_encoding_map)
print('')
print('TROCHUS')
print('File, Prediction')
for i in range(len(trochus_files)):
    print('{},{}'.format(trochus_files[i], predictions[i]))
print('')

## ############################################################################
## Repeat with ancillaria files
## ############################################################################
ancillaria_files = [
    './image_data/ancillaria/PAL-CL.01837o.png',
    './image_data/ancillaria/PAL-CL.01837n.png',
    './image_data/ancillaria/PAL-CL.01837m.png',
    './image_data/ancillaria/PAL-CL.01837l.png',
    './image_data/ancillaria/PAL-CL.01837k.png',
    './image_data/ancillaria/PAL-CL.01837j.png',
    './image_data/ancillaria/PAL-CL.01837i.png',
    './image_data/ancillaria/PAL-CL.01837h.png',
    './image_data/ancillaria/PAL-CL.01837g.png',
    './image_data/ancillaria/PAL-CL.01837f.png',
    './image_data/ancillaria/PAL-CL.01834b.png',
    './image_data/ancillaria/PAL-CL.01834a.png'
]
predictions = f.make_predictions(ancillaria_files, model, integer_encoding_map)
print('')
print('ANCILLARIA')
print('File, Prediction')
for i in range(len(ancillaria_files)):
    print('{},{}'.format(ancillaria_files[i], predictions[i]))
print('')

