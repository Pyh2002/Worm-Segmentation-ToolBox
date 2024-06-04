# Deep Learning Worm Segmentation User Guide

## Author

**Adam Smith** [adamfs@umich.edu](mailto:adamfs@umich.edu)

## Introduction

This tool utilizes deep learning methods, specifically convolutional neural networks, to address worm segmentation challenges. It is partly inspired by the [Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) and was developed by Adam Smith under the supervision of Eleni Gourgou at the University of Michigan. This tool is designed to effectively segment worms, especially when they are interacting with surrounding particles.

## Program Organization

### Model Training:

#### **data_organization.py:**

##### Dependencies:

1) os
2) cv2
3) random
4) shutil 

##### Setup:

1) Modify parameters in create_training_val_test on line 115

a)  First parameter is the directory of the original images 

b)  Second parameter is the directory of the masks

c)   Third parameter is the directory of the output 

d)  Fourth parameter is the number of validation image/mask pairs to be separated

e)  Fifth parameter is the number of test image/mask pairs to be separated

2) Ensure that the images are .png and ensure that the images and corresponding masks have the same file name

##### Run:

1) If set up correctly, the program will convert the masks to binary images and fill separate mask and image folders for train, validation, and test data sets
2) Ensure a proper split between train, test, and validation sets

#### **train_segmentation.py:**

##### Dependencies:

1) segmentation_models_pytorch
2) os
3) pytorch
4) torchvision
5) torchmetrics
6) tqdm
7) PIL

##### Setup:

1) Change training and validation mask/image file paths

a)  Lines 163-166

2) Adjust data loader parameters 

a)  Change num-workers to an appropriate number of cores for the computer on line 176

b)  Experiment with batch size so that the memory is not completely used up on line 175

3) Adjust other training parameter values that might influence results (optional)

a)  These values could include number of epochs, learning rate, input resize, etc. 

##### Run:

1) After correctly following the setup steps and running the program, ensure the program is training correctly

a)  The loss value should generally decrease if learning correctly

b)  Adjust parameters to improve training time or learning

2) At the end of training two state dictionaries of the model will saved

a)  best.pth corresponds to the state of the model with the best loss

b)  last.pth corresponds to the last state of the model 

### **Model Prediction:**

#### **main.py**

##### Dependencies:

1) os
2) cv2
3) pytorch
4) torchvision
5) segmentation_models_pytorch
6) numpy

##### Setup:

1) Set filepath to trained state dictionary on line 16
2) Set the directory for the images to be segmented on line 25
3) Set the output directory for the videos on line 35
4) Adjust any video output parameters if needed such as output size or framerate 
5) Uncomment line 63 to visualize results in real time if needed

##### Run:

1) If set up correctly, the console will print out each filename after it is finished generating the output

