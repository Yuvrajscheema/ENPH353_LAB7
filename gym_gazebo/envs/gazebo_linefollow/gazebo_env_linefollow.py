
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding

threshold = 70
subtract_constant = 20
class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return state, True

        # cv2.imshow("raw", cv_image)



        if not hasattr(self, "no_line_counter"):
            self.no_line_counter = 0

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        height, width = cv_image.shape[:2]
        bottom_height = int(0.3 * height)

        # Convert to RGB and separate channels to apply a custom filter.
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        red_rgb = frame_rgb.copy()
        red_rgb[:, :, 1] = 0
        red_rgb[:, :, 2] = 0

        blue_rgb = frame_rgb.copy()
        blue_rgb[:, :, 0] = 0
        blue_rgb[:, :, 1] = 0

        green_rgb = frame_rgb.copy()
        green_rgb[:, :, 0] = 0
        green_rgb[:, :, 2] = 0

        # Custom filter: emphasize yellow (red+green), suppress blue.
        filtered = (red_rgb.astype(np.float32) + green_rgb.astype(np.float32)) / 2 - blue_rgb.astype(np.float32)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        # Convert to grayscale and apply blur.
        frame_gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        blurred_gray = cv2.blur(frame_gray, (5, 5))

        # Thresholding logic to isolate the line.
        processed_gray = blurred_gray.copy()
        processed_gray[processed_gray > threshold] = 255
        processed_gray[processed_gray <= threshold] = np.clip(
            processed_gray[processed_gray <= threshold] - subtract_constant, 0, 255
        )

        # Focus on the bottom region of the image where the line is expected.
        subimage = processed_gray[-bottom_height:, :]

        # Calculate the weighted centroid of the line.
        y_rel, x = np.where(subimage < 255)
        if len(x) > 0:
            pixel_values = subimage[y_rel, x]
            weights = 255 - pixel_values
            cx = np.average(x, weights=weights)

            state_cx = min(int((cx / width) * 10), 9)
            state[state_cx] = 1
            self.no_line_counter = 0
        else:
            if self.no_line_counter > 30:
                done = True
            else:
                self.no_line_counter += 1
        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)



        # Set the rewards for your action
        if done:
            reward = -200
            return state, reward, done, {}

        if action == 0:
            if state[4] == 1 or state[5] == 1:
                reward = 4
            else:
                reward = 1
            return state, reward, done, {}

        if action == 1:
            if state[9] == 1:
                reward = 5
            elif state[8] == 1:
                reward = 4
            elif state[7] == 1:
                reward = 3
            elif state[6] == 1:
                reward = 2
            elif state[5] == 1:
                reward = 0
            else:
                reward = -10
            
            return state, reward, done, {}

        if state[0] == 1:
            reward = 5
        elif state[1] == 1:
            reward = 4
        elif state[2] == 1:
            reward = 3
        elif state[3] == 1:
            reward = 2
        elif state[4] == 1:
            reward = 0
        else:
            reward = -10

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
