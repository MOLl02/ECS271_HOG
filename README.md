# ECS271_HOG

Main package used is skimage, other used packages can be found in each file.

To run this code, first download the INRIA Person dataset from https://drive.google.com/file/d/16Wd_d-oIju7llItVI33BtlWkTFF1Q7EP/view?usp=drive_link

Then, to compute the HOG features, use the function slove() in hog.py. In this function, both custom_hog from custom_hog.py and hog function from skimage can be implemented.

In custom_hog.py and hog_network.py, there are functions to train SVM and Neural Network for the HOG feature to classify if the cropped image contains a person.

In windowpreprocess.py, there are functions to extract features about the bounding boxes and slide windows. The features contain the image size, bounding box size, and overlapping proportion of a sliding window with the object's bounding.

Using these features, neural networks to predict the bounding box size or overlapping proportion can be trained from window_network.py

With a trained classifier for person detection, function detect() from pyramid_img.py using slide windows can be used to detect persons on images of different sizes. 

Also, notebook test_customhog.ipynb can be easily ran to train the SVM of custom extracted HOG features.
