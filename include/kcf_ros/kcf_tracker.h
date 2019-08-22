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
#include <eigen_conversions/eigen_msg.h>

#include "kcf.h"

namespace kcf_ros
{
    class GaussianDistribution{
    public:
        int d = 2;
        Eigen::Vector2d mean = Eigen::Vector2d(0,0);
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(d, d);
        Eigen::MatrixXd cov_inverse = cov;
        double cov_det_sqrt = std::sqrt(cov.determinant());
    };

    class KcfTrackerROS : public nodelet::Nodelet
    {
    public:
        typedef message_filters::sync_policies::ExactTime<
            sensor_msgs::Image,
            kcf_ros::Rect
            > SyncPolicy;
        typedef message_filters::sync_policies::ApproximateTime<
            sensor_msgs::Image,
            kcf_ros::Rect
            > ApproximateSyncPolicy;

    protected:
        double pi = 3.141592653589793;

        int frames = 0;
        int callback_count_ = 0;
        bool debug_log_ = false;
        bool debug_view_ = false;
        double kernel_sigma_ = 0.5;
        double cell_size_ = 4;
        double num_scales_ = 7;
        bool is_approximate_sync_ = true;
        bool tracker_initialized_ = false;
        bool signal_changed_ = false;
        bool first_init_ = true;
        bool track_flag_ = false;
        int prev_signal_ = 0;
        int non_detected_count_ = 0;
        int offset_ = 0;
        int queue_size_ = 2;
        float detector_threshold_ = 100; // pixel
        float tracker_threshold_ = 100; // pixel
        int raw_image_width_ = 0;
        int raw_image_height_ = 0;

        int boxes_callback_cnt_ = 0;
        int prev_boxes_callback_cnt_ = 0;
        double detected_boxes_stamp_ = 0.0;
        int buffer_size_ = 100;

        KCF_Tracker tracker;
        autoware_msgs::DetectedObjectArray::ConstPtr detected_boxes_;

        std::vector<double> image_stamps;
        std::vector<cv::Mat> image_buffer;


        std::vector<BBox_c> tracker_results_queue_;
        std::vector<cv::Rect> detector_results_queue_;

        std_msgs::Header header_;

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;

        ros::Publisher debug_image_pub_;
        ros::Publisher croped_image_pub_;
        ros::Publisher output_rect_pub_;

        ros::Subscriber boxes_sub;

        boost::mutex mutex_;

        boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> > sync_;
        boost::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy> > approximate_sync_;
        message_filters::Subscriber<sensor_msgs::Image> sub_raw_image_;
        message_filters::Subscriber<kcf_ros::Rect> sub_nearest_roi_rect_;
        message_filters::Subscriber<autoware_msgs::DetectedObjectArray> sub_yolo_detected_boxes_;

        virtual void onInit();

        virtual void boxes_callback(const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes);

        virtual void callback(const sensor_msgs::Image::ConstPtr& raw_image_msg,
                              const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg);

        /* virtual void image_callback(const sensor_msgs::Image::ConstPtr& raw_image_msg); */

        virtual void visualize(cv::Mat& image,
                               const BBox_c& bb,
                               const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg,
                               double frames,
                               float box_movement_ratio);

        virtual void load_image(cv::Mat& image, const sensor_msgs::Image::ConstPtr& image_msg);

        virtual void publish_messages(const cv::Mat& image, const cv::Mat& croped_image,
                                      const BBox_c& bb, bool changed);

        virtual double calc_detection_score(const autoware_msgs::DetectedObject& box,
                                          const cv::Point2f& nearest_roi_image_center);

        virtual bool calc_gaussian(double& likelihood,
                                   const Eigen::Vector2d& input_vec,
                                   const GaussianDistribution& ditribution);

        virtual bool boxesToBox(const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes,
                                const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg,
                                cv::Rect& output_box,
                                float& score);

        virtual float check_detector_confidence(const std::vector<cv::Rect> detector_results,
                                                const float detection_score,
                                                float& movement,
                                                cv::Rect& init_box_on_raw_image);

        virtual float check_tracker_confidence(const std::vector<BBox_c> tracker_results);

        virtual bool enqueue_detection_results(const cv::Rect& init_box_on_raw_image);

        virtual bool enqueue_tracking_results(const BBox_c& bb);

    private:

    }; // class KcfTrackerROS

} // namespace kcf_ros

#endif
