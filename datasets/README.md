This folder contains directories with images (both original and augmented) along with their respective data.csv files in each directory. The data.csv we used had three main fields:
- image name
- linear_speed_x (which was kept constant throughout our data collection process)
- angular_speed_z

For a folder to be considered for training, it needs to contain the string '_YES'. This mechanism was to ensure that any other directories left unintentionally in the datasets folder were not automatically considered for training. 

The .csv file must be named data.csv for the image names and label names for that directory to be read from that file.

Note: The datasets folder in this repository contains a very small proportion of samples from the original datasets folder that we used, which was over 15 gigabytes. The purpose of this folder is only to represent the structure of what we used. The augment folder was produced as a result of the augmentation code that we ran on our original images.
