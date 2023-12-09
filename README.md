This repository contains the information needed to recreate our final project for Software Engineering for Machine Learning (Fall 2023)

Objective: Deploy a real model in a real safety-critical system. Autonomusly navigate a ROSbot XL in the hallways of Rice Hall avoiding obstacles.
Progress: Developed data collection and data augmentation pipeline. Trained several models uisng NVIDIA's Dave2 model template and deployed the models on the ROSbot XL. ROSbot currently is not able to fully navigate Rice Hall but can avoid walls and make left turns 20% of scenarios.
Future Steps: More data collection and model training to make our model more robust to various conditions (start position, lighting, navigating besides people in hallways)

Our project can be broken up into several steps:
1. Setting up the Husarion ROSbot XL, the ZED2 Camera, RPLIDAR, and XBOX One Controller
2. Data Collection and Augmentation
3. Model Training
4. Deployment


Team: Ishita Gupta, Nitin Maddi, Avaneen Pinninti  
Meriel von Stein's GitHub was used as a resource for this project: https://github.com/MissMeriel/ROSbot_data_collection/tree/master
