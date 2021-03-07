################################################################################
## Required imports
################################################################################
import shutil
from pathlib import Path
from PIL import Image, ImageOps
import os

################################################################################
## Helper functions
################################################################################
def resizeImage(desired_size, image_filepath):
    '''
    Takes a given image and desired size, then scales the image to that size
    while maintaining the aspect ratio.  Any excess space is filled with a
    black background.
    
    Example Usage: 
    resizeImage(224, '../image_data/fusus_longaevus/PAL-CL.01634a_D.png').show()
    '''
    image = Image.open(image_filepath)
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.BILINEAR)
    new_image = Image.new("RGB", (desired_size, desired_size))
    new_image.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_image

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