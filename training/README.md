


# Augmenting Data
We had two main augmentation files  ``augment.ipynb`` and ``augment_attempt_2.ipynb`` 
The first of which, ``augment.ipynb`` is to be used on images that have been split from the default camera output of left and right
This file will flip, blur and change the brightness of an image if the image was belonged to a turn. This first attempt at augmentation was 
a way of dealing with extremely unbalanced data, as we were driving in a completely straight path at this point of time. This helped balanced the data, 
but a slight drift would cause our model to have no chance of recovering

Our next augmentation file ``augment_attempt_2.ipynb`` was a supplement to our data collection methodology. After driving the S pattern, we saw that there were 
still many flaws that caused our robot to crash into the wall. While we were gathering more natural turn data, we were teaching the robot incorrect behavior. This is because during the S pattern, we would be turning into the wall, which was teaching the robot that turning into the wall was good. Instead, we wanted to teach it to turn the other way when looking at a wall, and to do this, we would have to negate the velocity while we were collecting data of the robot turning into the wall. So we decided to still drive the S pattern, but be able to recognize when the robot was turning towards the wall. To do this, we had an angular velocity (turn velocity) of magnitude 0.3 while driving towards the wall, and a magnitude of 0.8 when turning away from the wall. We were able to use this to negate the velocity while turning into walls, to tell our model to correct away from the wall rather than going toward a wall. This worked well in corridors, except at turns. The amount of turn data was negligible when compared to the overall amount of data, so the robot didn't learn to make turns well. To fix this issue, we implemented a trigger that would store a boolean indicating that our robot was doing a "real" turn rather than the S pattern. This allowed us to augment/multiply these turn images to help the model better learn these situations. 

Running these augmentation files will need you to:
1. Change the path to the dataset that you wish to augment
2. Change the name of the .csv file to data.csv, or change it in the path name
3. Set the path of where you want to save the augmented dataset. 


# DAVE2 Model Setup
Ishita


# Training a DAVE2 model
This portion of the code assumes that you have collected 


# Hyperparameters and Layers
Ishita

