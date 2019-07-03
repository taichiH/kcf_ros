#ifndef _KCF_ROS_KCF_TRACKER_
#define _KCF_ROS_KCF_TRACKER_

#include <stdlib.h>

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <cv_bridge/cv_bridge.h>
#include <kcf_ros/Rect.h>
#include <autoware_msgs/DetectedObjectArray.h>

#include "kcf.h"

namespace kcf_ros
{
  class KcfTrackerROS : public nodelet::Nodelet
  {
  public:
    typedef message_filters::sync_policies::ExactTime<
      sensor_msgs::Image,
      kcf_ros::Rect,
      autoware_msgs::DetectedObjectArray
      > SyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image,
      kcf_ros::Rect,
      autoware_msgs::DetectedObjectArray
      > ApproximateSyncPolicy;

  protected:
    int frames = 0;
    bool debug_log_ = false;
    bool debug_view_ = false;
    double kernel_sigma_ = 0.5;
    double cell_size_ = 4;
    double num_scales_ = 7;
    bool is_approximate_sync_ = true;

    KCF_Tracker tracker;

    std_msgs::Header header_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Publisher debug_image_pub_;
    ros::Publisher output_rect_pub_;

    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> > sync_;
    boost::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy> > approximate_sync_;
    message_filters::Subscriber<sensor_msgs::Image> sub_raw_image_;
    message_filters::Subscriber<kcf_ros::Rect> sub_nearest_roi_rect_;
    message_filters::Subscriber<autoware_msgs::DetectedObjectArray> sub_yolo_detected_boxes_;

    virtual void onInit();
    virtual void callback(const sensor_msgs::Image::ConstPtr& raw_image_msg,
                          const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg,
                          const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes);
    virtual void visualize(cv::Mat& image, const BBox_c& bb, double frames);
    virtual void load_image(cv::Mat& image, const sensor_msgs::Image::ConstPtr& image_msg);
    virtual void publish_messages(const cv::Mat& image, const BBox_c& bb, bool changed);
    virtual bool boxesToBox(const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes,
                            const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg,
                            cv::Rect& output_box);

  private:
  }; // class KcfTrackerROS
} // namespace kcf_ros

#endif
