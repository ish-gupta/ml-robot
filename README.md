This repository contains the information needed to recreate our final project for 

Objective:
Progress:
Future Steps: 

Our project can be broken up into several steps:

1. Setting up the Husarion ROSbot XL, the ZED2 Camera, RPLIDAR, and XBOX One Controller
2. Data Collection and Augmentation
3. Model Training
4. Deployment


The remainder of this README will contain ifnormation on how to setup the ROSbot XL and the sensors used for this project.
For information related to data collection and augmentation, model training, and deployment, please refer to their respective folders.

In order to setup the Husarion ROSbot XL these are the steps we followed:
1. Familiarized ourselves with ROS2 (https://docs.ros.org/en/humble/index.html). It is especially important to understand how to configure your environment, build packages, write nodes, and use topic visualization tools.
2. Follow the ROSbot XL quick start guide (https://husarion.com/tutorials/howtostart/rosbotxl-quick-start/). It is not mandatory to connect the camera and LIDAR at this stage. We recommend going through this guide making sure that by the end you can launch the basic ROSbot XL ROS2 nodes. One way to test this is if you can run the teleop_twist_keyboard node shown at the bottom of the quick start guide and your robot’s wheels move based on key presses.
We recommend connecting your ROSbot to an external display, mouse, and keyboard to setup the steps above. If a wifi connection is needed at any step above (we needed it to flash our ROSbot’s firmware), we recommend connecting to UVA Guest, writing down the MAC address associated with the wireless connection by running the 'ifconfig' command, and then registering that MAC address with UVA ITS so you can connect to the hidden ‘wahoo’ network. More information can be found here: https://virginia.service-now.com/its?id=itsweb_kb_article&sys_id=ca13d12bdb8153404f32fb671d961969. The 'ifconfig' command will tell you what the IP address is of your ROSbot when connected to various networks. 
If you’re able to complete step 2 without connecting to wifi, we still recommend that you connect to 'wahoo' on the ROSbot. This will allow you to ssh to the ROSbot so you can develop your programs remotely. We found using the 'nmtui' command to register the 'wahoo' network on the ROSbot followed by these two commands:
'nmcli device wifi rescan ssid wahoo'
'nmcli device wifi connect wahoo'
3. Once your ROSbot is properly configured, create your own ros2_ws folder. This is your colcon workspace where all of your development will occur. For more information on how to use 'colcon' to build packages see here: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html
4. In order to drive the ROSbot using the XBOX One Controller, we must connect the controller via bluetooth. More information on how to use the XBOX One Controller to actually drive the robot using the Joy package can be found in the Data Collection folder.









Team: Ishita Gupta, Nitin Maddi, Avaneen Pinninti