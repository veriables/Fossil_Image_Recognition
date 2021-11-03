###############################################################################
## This utility script scans a given 'folder' to find any subfolders within it
## which have less than 10 files.  If a subfolder is found that has less than
## 10 files, then that subfolder is moved from 'folder' to 'new_folder'.
##
## The reason this script exists is that the CNN isn't achieving very good 
## prediction accuracy (i.e., ~28%).  The root cause is a large number of 
## classes which only have 1-9 example images.  That just isn't enough 
## examples.  A typical image classification CNN would be given > 1000 images
## of each class.  So, in an effort to improve the data quality, we are 
## dropping classes that have fewer than 10 examples by moving them into a 
## different folder that won't be scanned during training and validation.
###############################################################################
import os
import shutil

folder = './image_data'
new_folder = './insufficient_images'


## Count the number of image files in each subdirectory of 'folder'
counts = []
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
for subdir in subfolders:
    image_count = 0
    for item in os.listdir(subdir):
        #print(item)
        if not item == '.DS_Store':
            if os.path.isfile(subdir + '/' + item):
                image_count += 1
    count = {
        "directory": subdir,
        "count": image_count
    }
    counts.append(count)

# If count is less than 10, move this subdirectory to a new location
for c in counts:
    if c['count'] < 10:
        source = c['directory']
        destination = new_folder + '/' + os.path.basename(os.path.normpath(c['directory']))
        print('{}  files   {}   --->   {}'.format(c['count'], source, destination))
        shutil.move(source, destination)

