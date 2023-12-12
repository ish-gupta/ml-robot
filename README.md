This repository contains the information needed to recreate our final project for Software Engineering for Machine Learning (Fall 2023)

Objective: Deploy a real model in a real safety-critical system. Autonomously navigate a ROSbot XL in the hallways of Rice Hall avoiding obstacles.
Progress: Developed data collection and data augmentation pipeline. Trained several models using NVIDIA's Dave2 model template and deployed the models on the ROSbot XL. Currently using 'old_model_360x640.pt' which can be found in the 'final' directory. Implemented protection protocal with the use of lidar to overwrite neural network predictions if they will cause a collision. ROSbot is currently able to make a full loop at times, but can get stuck in close corridors, specifically when dealing with non-convex shapes. Additional issues lie with right turns, and this robot has only seen success making a loop with all left turns. 
Future Steps: More data collection and model training to make our model more robust to various conditions (start position, lighting, navigating beside people in hallways)

Our project can be broken up into several steps:
1. Setting up the Husarion ROSbot XL, the ZED2 Camera, RPLIDAR, and XBOX One Controller
2. Data Collection and Augmentation
3. Model Training
4. Deployment


Team: Ishita Gupta, Nitin Maddi, Avaneen Pinninti  
Meriel von Stein's GitHub was used as a resource for this project: https://github.com/MissMeriel/ROSbot_data_collection/tree/master
