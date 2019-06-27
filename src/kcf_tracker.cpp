#include "kcf_ros/kcf_tracker.h"
#include "vot.hpp"

namespace kcf_ros
{
    void KcfTrackerROS::onInit()
    {
        nh_ = getNodeHandle();
        pnh_ = getPrivateNodeHandle();

        pnh_.getParam("debug_print", debug_log_);
        pnh_.getParam("debug_view", debug_view_);
        pnh_.getParam("kernel_sigma", kernel_sigma_);
        pnh_.getParam("cell_size", cell_size_);
        pnh_.getParam("num_scales", num_scales_);

        if (debug_log_) {
            std::cout << "kernel_sigma: " << kernel_sigma_ << std::endl;
            std::cout << "cell_size: " << cell_size_ << std::endl;
            std::cout << "num_scales: " << num_scales_ << std::endl;
        }

        frames = 0;
    }

    void KcfTrackerROS::visualize(cv::Mat& image, const BBox_c& bb, double frames)
    {
        cv::putText(image, "frame: " + std::to_string(frames),
                    cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::putText(image, "kernel_sigma: " + std::to_string(kernel_sigma_),
                    cv::Point(100, 200), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::putText(image, "cell_size: " + std::to_string(cell_size_),
                    cv::Point(100, 300), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::putText(image, "num_scales: " + std::to_string(num_scales_),
                    cv::Point(100, 400), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::rectangle(image, cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h),
                      CV_RGB(0,255,0), 2);

        if (debug_view_){
            cv::namedWindow("output", CV_WINDOW_NORMAL);
            cv::imshow("output", image);
            cv::waitKey();
        }
    }

    void KcfTrackerROS::load_image(cv::Mat& image, const sensor_msgs::Image::ConstPtr& image_msg)
    {
        try {
            cv_bridge::CvImagePtr cv_image =
                cv_bridge::toCvCopy(image_msg, "bgr8");
            image = cv_image->image;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("failed convert image from sensor_msgs::Image to cv::Mat");
            return;
        }
    }

    void KcfTrackerROS::publish_messages(const cv::Mat& image, const BBox_c& bb)
    {
        kcf_ros::Rect output_rect;
        output_rect.x = bb.cx;
        output_rect.y = bb.cy;
        output_rect.width = bb.w;
        output_rect.height = bb.h;
        output_rect_pub_.publish(output_rect);
        debug_image_pub_.publish(cv_bridge::CvImage(header_,
                                                    sensor_msgs::image_encodings::BGR8,
                                                    image).toImageMsg());
    }

    void KcfTrackerROS::callback(const sensor_msgs::Image::ConstPtr& image_msg,
                                 const kcf_ros::Rect::ConstPtr& rect_msg)
    {
        header_ = image_msg->header;
        cv::Mat image;
        load_image(image, image_msg);

        if (rect_msg->changed || frames == 0){
            if (debug_log_)
                ROS_WARN("init box !!!");
            tracker.init(image, cv::Rect(rect_msg->x,
                                         rect_msg->y,
                                         rect_msg->width,
                                         rect_msg->height));
        }


        double time_profile_counter = cv::getCPUTickCount();
        tracker.track(image);
        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
        if (debug_log_){
            float time = time_profile_counter / ((double)cvGetTickFrequency() * 1000);
            ROS_INFO("frame%d: speed-> %f ms/frame", frames, time);
        }

        BBox_c bb = tracker.getBBox();
        visualize(image, bb, frames);
        publish_messages(image, bb);

        frames++;
    }

} // namespace kcf_ros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(kcf_ros::KcfTrackerROS, nodelet::Nodelet)