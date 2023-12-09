import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge
import cv2
import numpy as np
import os


class DataCollectionNode(Node):
    def __init__(self):

        super().__init__('ros2_data_collection')
        self.subscription_twist = self.create_subscription(Twist, '/cmd_vel', self.vel_cmd_callback, 10)
        self.subscription_image = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.subscription_joystick = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        # self.subscription_image.add_on_set_parameters_callback(self.on_parameters_set).add_on_set_parameters_callback(self.on_parameters_set).add_on_change_callback(self.on_parameter_change)
                
        # self.subscription = self.create_subscription(Image, '/zed_camera/rgb/right_image', self.right_image_callback, 10)

        self.bridge = CvBridge()
        self.bridged_image = None
        self.image_num = 1
        
        self.linear_speed_x = 0.0
        self.angular_speed_z = 0.0
        self.is_turning = False

        self.dataset_subdir = "/media/husarion/Avaneen128/training_data_7"
        # self.dataset_subdir = "/media/husarion/Avaneen128/figure_eight"

        #find the smallest value that hasn't been published to, and make that the new data{num}.csv file
        self.collection_iter = 0
        self.dataset_path = self.dataset_subdir + "/data{}.csv".format(self.collection_iter)

        while os.path.exists(self.dataset_path):
            self.collection_iter += 1
            self.dataset_path = self.dataset_path = self.dataset_subdir + "/data{}.csv".format(self.collection_iter)

        with open(self.dataset_path, 'a') as f:
            f.write("{},{},{},{}".format("image name",
                                      "linear_speed_x",  
                                    "angular_speed_z",
                                    "is_turning"))
            f.write("\n")

        #timer callback to save the image and publish the velocities at that moment
        self.timer = self.create_timer(0.2, self.timer_callback)


    def timer_callback(self):
        if np.array_equal(self.bridged_image, None):
            return
        image_filename = "rivian{}-{:05d}.jpg".format(self.collection_iter, self.image_num) #:05d means 5 places
        cv2.imwrite("{}/{}".format(self.dataset_subdir, image_filename), self.bridged_image)

        with open(self.dataset_path, 'a') as f:
            f.write("{},{},{},{}".format(image_filename,
                                      self.linear_speed_x,
                                      self.angular_speed_z,
                                      self.is_turning))
            f.write("\n")
        self.image_num += 1 #will need to diff image numbers if we are doing left and right cameras

    def vel_cmd_callback(self, msg):
        self.linear_speed_x = msg.linear.x
        self.angular_speed_z = msg.angular.z

    
    def image_callback(self, msg):
        #might need to do some reversing, not too sure yet
        self.bridged_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
        print("Should get image here")
    
    def joy_callback(self, msg):
        if msg.buttons[7]:
            self.is_turning = True
        else:
            self.is_turning = False
        

    # def on_parameters_set(self, parameters):
    #     self.get_logger().info("Parameters set: {}".format(parameters))

    # def on_parameter_change(self, parameters):
    #     self.get_logger().info("Parameter change: {}".format(parameters))


    # def right_image_callback(self, msg):
    #     print("Should get right image here")


def main(args=None):
    print("hello mf")
    rclpy.init(args=args)
    node = DataCollectionNode()
    print("hola")

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
