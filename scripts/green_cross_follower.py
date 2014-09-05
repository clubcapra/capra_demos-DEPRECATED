import roslib; roslib.load_manifest('capra_demos')
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
import cv_bridge


def get_tshirt_mask(img):
    imgb = cv2.blur(img, (10, 10))
    hsv = cv2.cvtColor(imgb, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, np.array((140, 0, 0)), np.array((255, 255, 255)))
    thresh = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=3)
    thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=3)
    contour, hier = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(thresh,[cnt],0,255,-1)

    thresh = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=10)
    return thresh

def find_cross_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, np.array((55, 50, 0)), np.array((80, 255, 255)))
    return thresh

def find_cross_rectangle(img):
    mask = get_tshirt_mask(img)
    img = cv2.bitwise_and(img,img,mask = mask)
    thresh = find_cross_mask(img)

    small = cv2.resize(thresh, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("frame", small)
    cv2.waitKey(10)
    contour, hier = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    biggest_area = 0
    rect = (0, 0, 0, 0)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > biggest_area:
            rect = (x, y, w, h)
            biggest_area = area
    return rect


def image_callback(msg):
    bridge = cv_bridge.CvBridge()
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    print find_cross_rectangle(img)


rospy.init_node('green_cross_follower')
rospy.Subscriber('/scan', Image, image_callback)
rospy.spin()
cv2.destroyAllWindows()

