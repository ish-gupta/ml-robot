# This file imports the model weights & biases from a .pt file
# Uses the imported NN to steer the robot
# Also uses LIDAR to stop the robot when it is too close to an obstacle

import rclpy
from rclpy.node import Node

import os
import pandas as pd
import numpy as np
import sys
from PIL import Image
import cv2
from cv_bridge import CvBridge
import sensor_msgs.msg
from geometry_msgs.msg import Twist
import torch
from torchvision.transforms import Compose, ToTensor
from .DAVE2pytorch import DAVE2v3

class Steering_NN(Node):
    def __init__(self):

        super().__init__('steering_NN')

        # Load model weights & bias
        input_shape = (640, 360) # Input shape of the image
        self.model = DAVE2v3(input_shape=input_shape)
        # Change the path below based on where your .pt file is located on your robot
        self.model.load_state_dict(torch.load('/home/husarion/ros2_ws/src/final/final/model_30k_11epoch.pt', map_location=torch.device('cpu')))
        self.model.eval()

        # Creating velocity publisher and LIDAR 
        self.publisher_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.image_subscription = self.create_subscription(sensor_msgs.msg.Image, '/image_raw', self.image_callback, 10)
        self.lidar_subscription = self.create_subscription(sensor_msgs.msg.LaserScan, '/scan', self.lidar_callback, 10)
        print("lidar was subscribed to")

        self.min_pause_distance = 0.33 # If an obstacle is within 0.33 meters of the front of the robot, we stop
        self.obstacle_closeby = False

        self.bridge = CvBridge()
        self.bridged_image = None
        self.unsplit_image = None

        self.left_image = None
        self.right_image = None

        self.vel = Twist()
        self.vel.linear.x = 0.3 # Constant linear speed of the robot
        self.vel.angular.z = 0.0

        # Timer callback to publish the velocities at that moment
        self.timer = self.create_timer(0.2, self.timer_callback)


    def timer_callback(self):
        
        #UNSPLIT IMAGE
        # If an image hasn't been received by the callback or the robot is too close to an obstacle, do not use the NN to infer an angular velocity

        if self.unsplit_image == None or self.obstacle_closeby:
            return
        
        # Model inference to output angular velocity prediction
        transformed_image = Compose([ToTensor()])(self.unsplit_image)
        input_image = transformed_image.unsqueeze(0)
        self.vel.angular.z = self.model(input_image).item()

        self.publisher_vel.publish(self.vel)
        print(self.vel.angular.z)



        # SPLIT IMAGE CODE (used when model training and deployment used the left and right images from the ZED2 camera)
        '''
        if self.left_image == None or self.right_image == None or self.obstacle_closeby:
            return

        left_transformed_image = Compose([ToTensor()])(self.left_image)
        right_transformed_image = Compose([ToTensor()])(self.right_image)

        left_input_image = left_transformed_image.unsqueeze(0)
        right_input_image = right_transformed_image.unsqueeze(0)

        left_input_image = torch.autograd.Variable(left_input_image)
        right_input_image = torch.autograd.Variable(right_input_image)

        left_output = self.model(left_input_image)
        right_output = self.model(right_input_image)

        predicted_angular_velocity = (left_output.item()+right_output.item())/2
        self.vel.angular.z = predicted_angular_velocity

        print(self.vel.angular.z)

        self.publisher_vel.publish(self.vel)
        '''
    
    def image_callback(self, msg):
        self.bridged_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
        img = Image.fromarray(self.bridged_image)
        self.unsplit_image = img.resize((640,360))

        #SPLIT IMAGE CODE (used when model training and deployment used the left and right images from the ZED2 camera)
        '''
        img = Image.fromarray(self.bridged_image)
        width, height = img.size

        self.left_image = img.crop((0, 0, width // 2, height))
        self.right_image = img.crop((width // 2, 0, width, height))
        self.left_image = self.left_image.resize((320, 360))
        self.right_image = self.right_image.resize((320, 360))
        '''

    
    def lidar_callback(self, msg):
        lidar_ranges = msg.ranges
        front_indicies = len(lidar_ranges) // 6 #~60 degrees of points

        #Take the ranges from the lidar from the front
        front_ranges = lidar_ranges[-1 - front_indicies: -1] #the left 60 degrees from the middle 
        front_ranges.extend(lidar_ranges[0:front_indicies]) # the right 60 degrees from the middle
        
        close_counter = 0
        for curr_range in front_ranges:
            if curr_range < self.min_pause_distance or curr_range == float('inf'):
                close_counter += 1
        if close_counter > len(front_ranges) // 2.5:
            # If 40% of the points in front of the robot are within the stopping distance, stop the robot
            self.obstacle_closeby = True
            self.publisher_vel.publish(Twist()) # Stops the robot instantly
            #if this becomes false, we want the nn to predict on a new image, not some old stored one
            self.left_image = None
            self.right_image = None
            self.unsplit_image = None
            print("TOO CLOSE!!!")
        else:
            self.obstacle_closeby = False
            



def main(args=None):

    print("hello STEERING")
    rclpy.init(args=args)
    node = Steering_NN()
    print("bonjour")

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()