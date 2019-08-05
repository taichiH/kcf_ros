#!/usr/bin/env python

import rospy
import message_filters
from jsk_recognition_msgs.msg import RectArray, Rect
from jsk_recognition_msgs.msg import ClassificationResult
from autoware_msgs.msg import DetectedObjectArray, DetectedObject

class TransformMsgsJsk2Autoware():

    def __init__(self):
        self.pub = rospy.Publisher('~output', DetectedObjectArray, queue_size=1)

        queue_size = rospy.get_param('~queue_size', 1)

        sub_rects = message_filters.Subscriber(
            '~input_rects', RectArray, queue_size=queue_size)
        sub_classes = message_filters.Subscriber(
            '~input_classes', ClassificationResult, queue_size=queue_size)

        self.subs = [sub_rects, sub_classes]
        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)

        sync.registerCallback(self.callback)


    def callback(self, rects, classes):
        detected_object_array = DetectedObjectArray()
        detected_object_array.header = rects.header
        if len(rects.rects) != len(classes.label_names):
            rospy.logerr('rects and classes size is different')
            return

        for rect, label, score in zip(rects.rects, classes.label_names, classes.label_proba):
            detected_object = DetectedObject()
            detected_object.x = rect.x
            detected_object.y = rect.y
            detected_object.width = rect.width
            detected_object.height = rect.height
            detected_object.score = score
            detected_object.label = label
            detected_object_array.objects.append(detected_object)

        self.pub.publish(detected_object_array)


if __name__=='__main__':
    rospy.init_node('transform_msgs_jsk2autoware')
    transform_msgs_jsk2autoware= TransformMsgsJsk2Autoware()
    rospy.spin()
