## Fossil Image Recognition

### Welcome to the Project!
This project builds and uses a Convolutional Neural Network (CNN) to predict the scientific names of given fossil images.

### Features
The project has two main scripts:

1. **Build_Model.ipynb**: Creates and trains a CNN using a collection of images of fossils that have been labeled with their scientific name.
2. **Make_Predictions.ipynb**: The CNN will take, as input, a picture of a new fossil and give, as output, an identification of the fossil (i.e., the fossil's scientific name).

### Copyright Acknowledgement

#### Image source
The images of fossils were collected from the very excellent online palaeontology collection of the [Oxford University Museum of Natural History](https://oumnh.ox.ac.uk/collections-online#/search).  

The museum provides clear conditions for [online image use and copyright terms](https://oumnh.ox.ac.uk/collections-online-image-use-and-copyright-terms) and written permission was obtained for the use of their images in this project.

#### Implementation of copyright conditions
One condition of that permission is that every image displayed in this project must be labeled with an appropriate copyright acknowledge statement.

For images from [Oxford University Museum of Natural History](https://oumnh.ox.ac.uk/collections-online#/search), that copyright statement is:

<small>**Acknowledgement: &copy; Oxford University Museum of Natural History**</small>

For images from the [GB3D Type Fossils Project](http://www.3d-fossils.ac.uk/home.html) that copyright acknowledgement statement is:

<small>**Acknowledgement: &copy; Oxford University Museum of Natural History / www.3d-fossils.ac.uk / CC BY-NC-SA**</small>

This copyright statement is added as a physical part of each image used from the Oxford University Museum of Natural History.  Specifically, our copyright assurance procedure works like this:

(1) a picture of the copyright text is generated by a utility script (.src/utilities/Add_oxford_copyright_to_images.ipynb) in the appropriate size.

![](doc/images/copyright_acknowledgement.png)

(2) the generated picture of the copyright text is concatenated to the original image 

(3) to form a new picture which includes both images.  For example:

![](doc/images/PAL-CL.00370B.png)

This new picture, with the copyright acknowledgement "baked in", is the only image stored in or used by the project.  Thus, the copyright will be shown whenever the image is viewed and it will be carried along if any of the images or the project is copied by anyone.  Note: The preservation of these copyright statements is a condition of the license to use this project.

### Data Organization
To organize the image files, this project's "image_data" directory is divided into sub-directories that are each named after the scientific name of the specimens whose images they contain (e.g., all pictures of Fusus Longaevus are in a sub-folder named fusus_longaevus inside the image_data directory).  Thus, the image_data subfolders end up looking like:

![](doc/images/subfolder_organization.png)


To further organize the image data, the file names for each specimen match the Object Number used by the [Oxford University Museum of Natural History](https://www.oumnh.ox.ac.uk/).  Those Object Number can be found when viewing the image on the [Oxford University Musuem of Natural History online search page](https://www.oumnh.ox.ac.uk/collections-online#/search) where it is displayed near the top of the page as shown below.

![](doc/images/finding_object_numbers.png)

<small>*Acknowledgement: &copy; Oxford University Museum of Natural History*</small>

The fossils presented online from the [Oxford University Museum of Natural History's collection](https://www.oumnh.ox.ac.uk/) are often mounted so several fossils are included in each photograph.  Like the image below:

![](doc/images/fusus_uniplicatus_PAL-CL.01712c.png)

<small>*PAL-CL.01712c - Acknowledgement: &copy; Oxford University Museum of Natural History*</small>

In order to present the CNN with examples of individual specimens, we cut these multi-specimen images into an image per specimen.  To continue using the example from above; it became the following three images.

![](doc/images/PAL-CL.01712c_A.png)

<small>*PAL-CL.01712c - Acknowledgement: &copy; Oxford University Museum of Natural History*</small>

![](doc/images/PAL-CL.01712c_B.png)

<small>*PAL-CL.01712c - Acknowledgement: &copy; Oxford University Museum of Natural History*</small>

![](doc/images/PAL-CL.01712c_C.png)

<small>*PAL-CL.01712c - Acknowledgement: &copy; Oxford University Museum of Natural History*</small>

Note: As you can see, these sliced images were "stamped" with a copyright acknowledgement using a utility script which is included in the project at src/utilities/Add_oxford_copyright_to_images.ipynb.

These sliced images were then given file names that start with their collection object number (e.g., PAL-CL.01712c) and, then, are suffixed with an underscore and the letter designation written next to that specimen on the card to which they were mounted in the photograph.  For example, the image above was given the filenames **PAL-CL.01712c_A.png**, **PAL-CL.01712c_B.png**, and **PAL-CL.01712c_C.png** respectively.

Here is an example of how the folder structure and file naming standard combine to look within the project's image_data directory.

![](doc/images/directory_structure_example.png)

This directory structure is convenient for general organization, but more importantly, the CNN will use the folder names as class names for all the examples contained within each folder.  So, it will understand that everything in the fusus_uniplicatus folder is a picture of a Fusus Uniplicatus fossil.

### Make it Challenging
Two challenges are immediately obvious and should be addressed within the first iteration of this project.  

1. **Rare examples**: For many fossils, we don't have many specimens (sometimes only 30 or less).  This is nowhere near as many examples as are often used in training CNNs (i.e., 1000+).
2. **Similar examples**: Some fossils are remarkably similar to each other even when they have been assigned to different species.

To ensure we start on the right foot, our first iteration of the CNN will distinguish fusus_uniplicatus (which has 30 specimens) from fusus_longaevus (which has 28 specimens).

As you can see, they are pretty similar.

![](doc/images/PAL-CL.01698a_C.png)

<small>Fusus Longaevus - PAL-CL.01712c - Acknowledgement: &copy; Oxford University Museum of Natural History</small>

![](doc/images/PAL-CL.01712c_A.png)

<small>Fusus Uniplicatus - PAL-CL.01712c - Acknowledgement: &copy; Oxford University Museum of Natural History</small>


### Prerequisites
This project uses tensorflow and you will need to have installed the tensorflow module version 2.4.1 or higher.

You can check if you have tensorflow (and its version) from the command line with:
Linux/Mac:
```
pip freeze | grep tensorflow
```
Windows:
```
pip freeze | findstr tensorflow
```

If you are missing tensorflow, you can install it with:
```
pip install tensorflow
```

If you have tensorflow, but it needs updating, that task can be accomplished with adding the -U (update) flag to the install command like:
```
pip install tensorflow -U
```

### Amazon Web Services Instance Setup
In order to build the model in a reasonable amount of time, we need to leverage GPU support.  One widely accessible way to allow many people to use this codebase on GPUs is to include instructions for setting up an Amazon Web Services EC2 host to run the Build_Model.py script.  Here are the steps:

1. Sign up for an Amazon Web Services account
2. Launch a new EC2 instance:
    a. Select an AMI: the Deep Learning AMI (Amazon Linux 2) Version 52.0 - ami-0911c789823a92ffa
    b. Select an Instance Type: p3.8xlarge ($12.24/hr)
    c. Configure Instance Details: Leave defaults
    d. Add Storage: Change the default 110 GiB to 1023 GiB
    e. Add Tags: Leave Defaults
    f. Configure Security Groups: Create a new security group.  Edit the SSH security rule to only allow access from your current IP address (e.g., SSH TCP 22 FROM 75.118.37.95/32 )
    g. Create a new KeyPair and download it to your local machine
3. Once the instance is launched record the Public IPv4 DNS address (e.g., ec2-54-213-208-83.us-west-2.compute.amazonaws.com)
4. Connect to your instance
    a. Change the permissions on the downloaded KeyPair file to allow it to be used by ssh:
        ```
        cd ~/Downloads/;
        chmod 0400 /Users/me/Downloads/fossils_002.pem;
        ```
    b. Connect to your instance via ssh
        ```
        ssh -i /Users/me/Downloads/fossils_002.pem ec2-user@ec2-54-213-208-83.us-west-2.compute.amazonaws.com
        ```
        
### How to Build the Model
1. Activate the tensorflow_p37 environment
    ```
    source activate tensorflow_p37;
    ```
2. Download the project from github
    ```
    https://github.com/veriables/Fossil_Image_Recognition.git
    ```
3. Run the Build_Model.py script
    ```
    cd project/src;
    python ./Build_Model.py;
    ```
4. When the script finishes, it will output the best accuracy it obtained as well as the locations of the *saved model file* and the *class_labels_map file*.  These two files are the inputs for the Make_Predictions.py script

5. If desired, you can download the model and class_labels_map files by running commands like these on your local machine:
    ```
    cd ~/Downloads;

    scp -i /Users/jp/Downloads/fossils_002.pem ec2-user@ec2-54-213-208-83.us-west-2.compute.amazonaws.com:/home/ec2-user/for_aws2/model/best_model_2021-11-03-07-15-16.hdf5 .;

    scp -i /Users/jp/Downloads/fossils_002.pem ec2-user@ec2-54-213-208-83.us-west-2.compute.amazonaws.com:/home/ec2-user/for_aws2/class_labels/labels_map_2021-11-03-07-15-16.pkl .;
    ```
### How to Use the Model
After you have built a model, you can upload pictures to your EC2 instance using SCP.  For example:
```
scp -i /Users/me/Downloads/fossils_002.pem /Users/jp/Desktop/AAAAAA_Regorg/projects/Fossil_Image_Recognition_GPU/images.zip ec2-user@ec2-54-213-208-83.us-west-2.compute.amazonaws.com:/home/ec2-user/images.zip;
```

Then, extract them with:
```
unzip images.zip;
```

Now, you can edit the Make_Predictions.py script to 
1. List your files in an array
2. Set the path to the saved model file
3. Set the path to the saved class label map file

And, then you can simply run the Make_Predictions.py script to generate a list of predicted fossil names for each of the given image files.
```
python ./Make_Predictions.py
```

### The Insufficient Images directory
Although we have catalouged over 373 different species of fossil, many of these species have too few example images to reliably train a neural network.  For example, 144 species have only one image!  We arbitrarily decided on a cutoff of 10 examples, then moved the others from the 'image_data' directory into a directory named 'insufficient_images'.  We use 'image_data' to train and validate the neural network.  We use the 'insufficient_images' directory to keep collecting more images so that some day, when a species has enough examples, we can move it over to the 'image_data' directory and include it in training the neural network.


## How to Contribute

At the start of this project, we have created a Convolutional Neural Network that classifies images a few of types of fossils proving it can handle some of the obvious challenges.  There are, of course, many improvements that could be made.  You could submit pull requests that cut redundancies, adjust preprocessing parameters, experiment with changing optimizers, tune hyperparameters, add layers to the model, or (very importantly) add images of fossils, etc.

## Credits

**Oxford University Museum of Natural History** (https://oumnh.ox.ac.uk/collections-online#/search): For providing such a wonderful dataset of fossils with pictures and taxonomy data.  All the images used to train this CNN are &copy; Oxford University Museum of Natural History and marked as such wherever they are stored, transmitted, processed (yes - the CNN has to learn to deal with those copyright stamps), or displayed.  These copyright notices must be preserved if the project is copied or forked.






