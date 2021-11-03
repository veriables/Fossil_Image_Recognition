################################################################################
## Required imports
################################################################################
import shutil
from pathlib import Path
from PIL import Image, ImageOps
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from tensorflow.keras.models import Sequential
from PIL import Image, ImageOps
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import zoom

################################################################################
## Functions
################################################################################
def getImagesAndClasses(path):
    '''
       Given a path to a directory that contains subfolders each containing images
       of a particular class (note: the subfolder names are the class names), this
       function will return three arrays.
       1. image_data : An array of ndarrays that contain the image data (uint8)
       2. class_names: An array of class names that map to the corresponding elements in the image array
       3. file_paths : An array of paths to the original image files
    '''
    file_paths = []
    class_names = [] # the subdirectory name in which the image was located
    image_data = []
    
    # Step through all the subdirectories and their files
    # to get each filename and it's class name
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if not 'DS_Store' in file:
                file_paths.append(os.path.join(subdir,file))
                class_names.append(subdir[subdir.rindex('/')+1:])

    counter = 0
    for counter in range(len(file_paths)):
        image = Image.open(file_paths[counter])
        image_arr = np.asarray(image)
        image_data.append(image_arr)
        counter += 1

    return image_data, class_names, file_paths


def resizeImage(desired_size, image_filepath):
    '''
    Takes a given image and desired size, then scales the image to that size
    while maintaining the aspect ratio.  Any excess space is filled with a
    black background.
    '''
    image = Image.open(image_filepath)
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.BILINEAR)
    new_image = Image.new("RGB", (desired_size, desired_size))
    new_image.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_image


def resize_image(image_ndarray, desired_size):
    '''
    Takes a given image and desired size, then scales the image to that size
    while maintaining the aspect ratio.  Any excess space is filled with a
    black background.
    
    Example Usage: 
    resizeImage(img_arr, 224).show()
    '''
    image = Image.fromarray(image_ndarray) # <-- error occurs here
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.BILINEAR)
    new_image = Image.new("RGB", (desired_size, desired_size))
    new_image.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    output_image_ndarray = np.array(new_image)
    return output_image_ndarray
    
    
def normalize_image(img):
    '''
    Convert the given image which has pixel values in the range 0-255 and
    adjust those values so they lie in the range 0-1.
    '''
    img = img.astype('float32')
    img /= 255.0
    return img


def expand_with_rotated_images(images, labels, num_copies):
    '''
       Take the original arrays of images and labels and add instances of each 
       image each with different rotations.  Return the data as two lists (images_list
       and labels_list)
    '''
    images_list = []
    labels_list = []
    if num_copies <=0:
        num_copies = 0

    for i in range(len(images)):
        original_img = images[i]
        original_label = labels[i]
        images_list.append(original_img)
        labels_list.append(original_label)
        for j in range(num_copies):
            images_list.append(rotateImage(original_img, 0, 180))
            labels_list.append(original_label)
    return images_list, labels_list


def createTemporaryDirectory(source_directory):
    '''
    Takes a source_directory name and creates a copy of that directory and all
    its contents saving it as a directory named the same as the given source
    directory, but with the prefix "temp_"
    '''
    head, tail = os.path.split(source_directory)
    destination_directory = os.path.join(head, 'temp_' + tail)
    destination = shutil.copytree(source_directory, destination_directory)
    
    
def removeTemporaryDirectory(directory_to_remove):
    '''
    Takes a directory name and traverses the directory recursively
    removing every subdirectory and file found under it.  Then, it
    removes the directory itself.
    '''
    shutil.rmtree(directory_to_remove)
    
    
def resizeImagesInDirectory(directory_name):
    '''
    Iterate through all the files in this directory recursively.
    Open each png image file, resizes it, and saves it overwriting 
    the original version.
    '''
    for subdir, dirs, files in os.walk(directory_name):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".png"):
                resizedImage = resizeImage(224, filepath)
                resizedImage.save(filepath)
                

def rotateImage(image, minDegrees = 0, maxDegrees = 180):
    '''
       Rotates a given image by a random amount between a given minimum
       and maximum number of degrees.
    '''
    rotation = random.randint(minDegrees,maxDegrees)
    return rotate(image, angle=rotation)

    
def saveImage(filepath, image):
    '''
    Saves an image to file
    '''
    image.save(filepath)
    
    
def createDirectoryIfNecessary(path):
    '''
    Checks if the given directory path exists.  If it does not exist,
    then it is created.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def saveImages(basepath, images, labels):
    '''
    Watch out for: KeyError: ((1, 1, 3), '<f4')
    The error message seems to be complaining about the shape, but it is really about the data type. Multiplying by 255 and then changing to uint8 fixes the problem.
    '''
    createDirectoryIfNecessary(basepath)
    for i in range(len(images)):
        img_array = np.multiply(images[i], 255)
        typed_array = img_array.astype(np.uint8)
        image = Image.fromarray(typed_array)
        label = labels[i]
        createDirectoryIfNecessary(os.path.join(basepath,label))
        filepath = os.path.join(basepath, label, str(i) + '.png')
        #print('filepath: {}'.format(filepath))
        saveImage(filepath, image)


def get_integer_map(labels_list):
    '''
    The index of the returned list will be the integer representation of this label
    '''
    return list(set(labels_list))
    
    
def convert_to_one_hot(inputIndex, mapObj):
    '''
    Creates a one-hot array of all zeros, except that at the given inputIndex which
    has a one.
    '''
    output = np.zeros(len(mapObj))
    output[inputIndex] = 1
    return output


def decode_class_name(numpy_onehot_value, integer_encoding_map):
    '''
    Class names are text values (e.g., auricula_ringens) which are not useful for a 
    numeric engine like a neural network.  So, we gave each of them a numeric 
    replacement value.  However, that is still not ideal because the neural network
    could assume that there is a relationship in those numbers and, in our case,
    these numbers are just categorical labels (i.e., the fact that ancillaria is labeled
    1 and marginella is labeled 3 doesn't mean that ancillaria is lower on any scale).
    So, we next convert our integer labels into one-hot encoded values.
    
    Now the problem is going all the way back from one-hot encoded values (i.e., which
    is the form in which our prediction will appear) through the integer label and,
    finally, back to the class name (which we can easily read).  That's what this
    function does.
    '''
    index = np.argmax(numpy_onehot_value.tolist())
    class_name = integer_encoding_map[index]
    return class_name


def make_predictions(list_of_filepaths, model, integer_encoding_map):
    '''
    This function processes each image from list_of_filepaths in the same way 
    the training images were processed.  Then, it passes each image to the model's
    predict method.  Finally, it uses the integer_encoding_map to decode each 
    integer label into a class name for that fossil.  It then returns a list of 
    predicted class names.
    '''
    predicted_class_names = []
    for i in list_of_filepaths:
        image_filepath = i
        image = Image.open(image_filepath)
        image_arr = np.asarray(image)
        image = np.array(resize_image(np.array(image), 224))
        image = np.array(normalize_image(image))
        img = np.expand_dims(image, axis=0)
        prediction_output = model.predict(img)
        predicted_class_name = decode_class_name(prediction_output, integer_encoding_map)
        predicted_class_names.append(predicted_class_name)
    return predicted_class_names

