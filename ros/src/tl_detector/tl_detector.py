#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

#Addiyional Libs
import numpy as np
from datetime import datetime
import os
import math

STATE_COUNT_THRESHOLD = 3

#cv2.namedWindow("Image", 0)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.slp = self.config['stop_line_positions'] # stop line positions

        self.nslp = lambda: expando
        self.nslp.position = lambda: expando
        self.nslp.position.x = 0
        self.nslp.position.y = 0

        self.imH = self.config['camera_info']['image_height'] # Image height
        self.imW = self.config['camera_info']['image_width'] # Image Width
        self.flY = self.config['camera_info']['focal_length_y'] # focal length Y
        self.flX = self.config['camera_info']['focal_length_x'] # focal Length X

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        directory = './images/'



        print('Total of ', len(self.slp), ' stop lines')

        # Create the folder to store saved images
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.bridge = CvBridge()
        self.initDet = False
        print
        print('---\tWaiting for detector to initialize\t---')
        print
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.light_classifier = TLClassifier()
        self.initDet = True
        self.listener = tf.TransformListener()



        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        #print('Total of ', len(self.waypoints.waypoints), ' waypoints')

    def traffic_cb(self, msg):
        self.lights = msg.lights
        #print('Total of ',len(self.lights),' traffic lights')
        #Get state of traffic lights

        #print(len(self.lights))
        #for i in range(0, len(self.lights)):
        #    print(i,self.lights[i].state)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()
        #print(light_wp,state)
        #Save Images in <images> folder
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = np.array(frame, dtype=np.uint8)
        filename = './images/'+str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '___' + str(state) + '___.jpg'
        #cv2.imwrite(filename,frame)
        



        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        #print(pose.position.x,pose.position.y)

        dist = float('Inf')
        dist_i = 0
        for i in range(0, len(self.waypoints.waypoints)):
            way_pos_x = self.waypoints.waypoints[i].pose.pose.position.x
            way_pos_y = self.waypoints.waypoints[i].pose.pose.position.y
            d = math.sqrt(math.pow(pose.position.x+5 - way_pos_x, 2) + math.pow(pose.position.y - way_pos_y,2))
            if (d < dist):
                dist = d
                dist_i = i

        #print(dist_i,self.waypoints.waypoints[dist_i].pose.pose.position.x, self.waypoints.waypoints[dist_i].pose.pose.position.y)

        return dist_i


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        #print(point_in_world)
        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image or self.initDet == False):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if(len(self.lights) == 0 or self.initDet == False):
            return -1, TrafficLight.UNKNOWN

        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            #print
            #print('Nearest waypoint to vehicle:')
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

        #find closest traffic light
        #print(self.pose.pose.position.x,self.pose.pose.position.y)

        dist = float('Inf')
        dist_i = 0
        for i in range(0,len(self.lights)):
            obst_pos_x = self.lights[i].pose.pose.position.x
            obst_pos_y = self.lights[i].pose.pose.position.y
            #print(obst_pos_x,obst_pos_y)
            d = math.sqrt(math.pow(self.pose.pose.position.x - obst_pos_x, 2)+math.pow(self.pose.pose.position.y - obst_pos_y, 2))
            if(d < dist and self.pose.pose.position.x < obst_pos_x):
                dist = d
                dist_i = i

        #print(dist,dist_i,self.lights[dist_i].pose.pose.position.x,self.lights[dist_i].pose.pose.position.y)


        # find stop line closest to the traffic light

        obst_pos_x = self.lights[dist_i].pose.pose.position.x
        obst_pos_y = self.lights[dist_i].pose.pose.position.y

        light = self.lights[dist_i]

        dist = float('Inf')
        dist_i = 0
        for i in range(0, len(self.slp)):
            d = math.sqrt(math.pow(self.slp[i][0] - obst_pos_x, 2) + math.pow(self.slp[i][1] - obst_pos_y,2))
            if (d < dist and self.slp[i][0] < obst_pos_x):
                dist = d
                dist_i = i

        self.nslp.position.x = self.slp[dist_i][0]
        self.nslp.position.y = self.slp[dist_i][1]

        dist = math.sqrt(math.pow(self.pose.pose.position.x +5 - self.nslp.position.x, 2) + math.pow(self.pose.pose.position.y - self.nslp.position.y, 2))
        #print(dist)
        #if (dist > 50):
        #    return -1, TrafficLight.UNKNOWN
        #else:
        #    print('*** Approaching Traffic Light ***')

        txt=[]
        #print('Nearest waypoint to stop line:')
        light_wp = self.get_closest_waypoint(self.nslp)
        #print(light_wp)

        if light:
            state = self.get_light_state(light)
            #print("Ground Truth: ",self.lights[dist_i].state, " - Classification: ", state)

            #frame = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            #filename = './images_res/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '___' + str(self.lights[dist_i].state) + '-' + str(state) + '___.jpg'
            #cv2.imwrite(filename,frame)

            #state = self.lights[dist_i].state

            if(state == TrafficLight.RED):
                return light_wp, state
            #    print('go')
            #else:
            #    print('stop')

        #self.waypoints = None # This was commented out on purpose

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
