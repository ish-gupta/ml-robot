## Setup

This README will contain information on how to setup the ROSbot XL and the sensors used for this project.
For information related to data collection and augmentation, model training, and deployment, please refer to their respective folders.

In order to setup the Husarion ROSbot XL these are the steps we followed:
1. Familiarized ourselves with ROS2 (https://docs.ros.org/en/humble/index.html). It is especially important to understand how to configure your environment, build packages, write nodes, and use topic visualization tools.
2. Follow the ROSbot XL quick start guide (https://husarion.com/tutorials/howtostart/rosbotxl-quick-start/). It is not mandatory to connect the camera and LIDAR at this stage. We recommend going through this guide making sure that by the end you can launch the basic ROSbot XL ROS2 nodes. One way to test this is if you can run the `teleop_twist_keyboard` node shown at the bottom of the quick start guide and your robot’s wheels move based on key presses.
We recommend connecting your ROSbot to an external display, mouse, and keyboard to setup the steps above. If a wifi connection is needed at any step above (we needed it to flash our ROSbot’s firmware), we recommend connecting to UVA Guest, writing down the MAC address associated with the wireless connection by running the `ifconfig` command, and then registering that MAC address with UVA ITS so you can connect to the hidden **wahoo** network. More information can be found here: https://virginia.service-now.com/its?id=itsweb_kb_article&sys_id=ca13d12bdb8153404f32fb671d961969. The `ifconfig` command will tell you what the IP address is of your ROSbot when connected to various networks. 
If you’re able to complete step 2 without connecting to wifi, we still recommend that you connect to wahoo on the ROSbot. This will allow you to ssh to the ROSbot so you can develop your programs remotely. We found using the `nmtui` command to register the wahoo network on the ROSbot followed by these two commands:
`nmcli device wifi rescan ssid wahoo`
`nmcli device wifi connect wahoo`
3. Once your ROSbot is properly configured, create your own ros2_ws folder. This is your colcon workspace where all of your development will occur. For more information on how to use `colcon` to build packages see here: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html
4. In order to drive the ROSbot using the XBOX One Controller, we must connect the controller via bluetooth.
You should have your Xbox controller's MAC address before you begin. The easiest way to find it out is to connect it to a bluetooth-enabled laptop and inspect the device using your bluetooth settings.
You should also have `bluez` already installed on your ROSbot. Find out by running `bluetoothctl`. If not, run `sudo apt install bluetoothctl`.
Run `sudo service bluetooth restart; bluetoothctl`. This will take you into the bluetoothctl prompt.
Within the bluetoothctl prompt, run `remove all`
Within the bluetoothctl prompt, run `bluetoothctl scan on`
Within the bluetoothctl prompt, run `connect <your-controller-MAC>`
You should see output similar to ``. If not, make sure that the XBOX One Controller is in pairing mode (quickly flashing white light).
Once the XBOX One Controller is connected to the ROSbot, you can run `ros2 run joy joy_node` which will parse input from the controller and publish it to the `/joy` node. The `drive` node shows how we use `/joy` messages to move the robot. Joy package documentation can be found here: https://index.ros.org/p/joy/
5. The ZED2 camera is a stereo camera that has is utilized in many applications that require depth and AI sensing. We tried to use the supported ROS2 packages that are available for the ZED2 but ran into many issues due to our ROSbot not having an NVIDIA GPU. Our solution was to use the `usb_cam` package to parse the input of the ZED2 camera. We installed the `usb_cam` package into our colcon workspace (https://github.com/ros-drivers/usb_cam/tree/ros2). You'll see that you can run the camera node using the command `ros2 run usb_cam usb_cam_node_exe`. This command takes in as input a .yaml file to collect images from the camera and this .yaml file needs to be edited to match the camera we have. For our ROSbot, the .yaml file was stored in this directory `/home/husarion/.ros/camera_info/default_cam.yaml`. We edited this .yaml file to what is seen in `default_cam.yaml` and ran the command `ros2 run usb_cam usb_cam_node_exe --ros-args --params-file /home/husarion/.ros/camera_info/default_cam.yaml`. The image data is published to the `/image_raw` topic and is of type `sensor_msgs/Image`. Note: `.ros` is a hidden folder.
6. [LIDAR SETUP]



Team: Ishita Gupta, Nitin Maddi, Avaneen Pinninti
Meriel von Stein's GitHub was used as a resource for this project: https://github.com/MissMeriel/ROSbot_data_collection/tree/master
