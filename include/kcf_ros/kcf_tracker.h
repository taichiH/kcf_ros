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

    class ImageInfo {
    public:
      double stamp = 0.0;
      int signal = 0;
      cv::Mat image;
      cv::Rect rect;

      bool initialized = false;

      explicit ImageInfo(const cv::Mat& _image = cv::Mat(3,3,CV_8UC3),
                         const cv::Rect& _rect= cv::Rect(0,0,0,0),
                         const int _signal = 0,
                         const double _stamp = 0.0) {
          stamp = _stamp;
          signal = _signal;
          image = _image;
          rect = _rect;
      }
      ~ImageInfo() {};

      void init(const cv::Mat& _image = cv::Mat(3,3,CV_8UC3),
                const cv::Rect& _rect= cv::Rect(0,0,0,0),
                const int _signal = 0,
                const double _stamp = 0.0) {
          initialized = true;
          stamp = _stamp;
          signal = _signal;
          image = _image;
          rect = _rect;
      }

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

        typedef std::shared_ptr<ImageInfo> ImageInfoPtr;

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
        int interpolation_frequency_ = 1;

        bool tracker_initialized_ = false;
        bool track_flag_ = false;

        bool signal_changed_ = false;
        int signal_ = 0;
        int prev_signal_ = 0;

        int offset_ = 0;
        int queue_size_ = 2;

        float box_movement_thresh_ = 0.02;

        int raw_image_width_ = 0;
        int raw_image_height_ = 0;

        int cnt_ = 0;
        int boxes_callback_cnt_ = 0;
        int prev_boxes_callback_cnt_ = 0;
        double detected_boxes_stamp_ = 0.0;
        int buffer_size_ = 100;

        KCF_Tracker tracker;
        /* ImageInfo image_info_; */

        autoware_msgs::DetectedObjectArray::ConstPtr detected_boxes_;

        std::vector<double> image_stamps;
        std::vector<cv::Mat> image_buffer;
        std::vector<cv::Rect> rect_buffer;


        std::vector<cv::Rect> tracker_results_buffer_;

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
                               const cv::Rect& rect,
                               const cv::Rect& nearest_roi_rect,
                               double frames,
                               float box_movement_ratio = 0,
                               float tracker_conf = 0,
                               float tracking_time = 0,
                               std::string mode = "");

        virtual void load_image(cv::Mat& image, const sensor_msgs::Image::ConstPtr& image_msg);

        virtual void publish_messages(const cv::Mat& image, const cv::Mat& croped_image,
                                      const cv::Rect& rect, bool changed);

        virtual bool calc_gaussian(double& likelihood,
                                   const Eigen::Vector2d& input_vec,
                                   const GaussianDistribution& ditribution);

        virtual bool boxesToBox(const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes,
                                const cv::Rect& roi_rect,
                                cv::Rect& output_box,
                                float& score);

        virtual bool get_min_index(int& min_index);

        virtual bool box_interpolation(int min_index);

        virtual bool create_buffer(const ImageInfoPtr& image_info);

        virtual bool clear_buffer();

        virtual bool create_tracker_results_buffer(const cv::Rect& bb);

        virtual float check_confidence(const std::vector<cv::Rect> results,
                                       float& box_movement_ratio);

        virtual bool update_tracker(cv::Mat& image, cv::Rect& output_rect, const cv::Rect& roi_rect,
                                    float& box_movement_ratio, float& tracker_conf, float& tracking_time);

        virtual double calc_detection_score(const autoware_msgs::DetectedObject& box,
                                          const cv::Point2f& nearest_roi_image_center);

        virtual void increment_cnt();

        virtual int calc_offset(int x);

        virtual float sigmoid(float x, float a=200);

    private:
    }; // class KcfTrackerROS



} // namespace kcf_ros

#endif
