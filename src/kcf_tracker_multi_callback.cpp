#include "kcf_ros/kcf_tracker.h"

namespace kcf_ros
{
    void KcfTrackerROS::onInit()
    {
        nh_ = getNodeHandle();
        pnh_ = getPrivateNodeHandle();

        pnh_.getParam("debug_log", debug_log_);
        pnh_.getParam("debug_view", debug_view_);
        pnh_.getParam("kernel_sigma", kernel_sigma_);
        pnh_.getParam("cell_size", cell_size_);
        pnh_.getParam("num_scales", num_scales_);
        pnh_.getParam("approximate_sync", is_approximate_sync_);
        pnh_.getParam("offset", offset_);

        if (debug_log_) {
            std::cout << "kernel_sigma: " << kernel_sigma_ << std::endl;
            std::cout << "cell_size: " << cell_size_ << std::endl;
            std::cout << "num_scales: " << num_scales_ << std::endl;
        }

        debug_image_pub_ = pnh_.advertise<sensor_msgs::Image>("output_image", 1);
        croped_image_pub_ = pnh_.advertise<sensor_msgs::Image>("output_croped_image", 1);
        output_rect_pub_ = pnh_.advertise<kcf_ros::Rect>("output_rect", 1);

        image_sub =
            pnh_.subscribe("input_raw_image", 1, &KcfTrackerROS::image_callback, this);

        sub_nearest_roi_rect_.subscribe(pnh_, "input_nearest_roi_rect", 1);
        sub_yolo_detected_boxes_.subscribe(pnh_, "input_yolo_detected_boxes", 1);

        if (is_approximate_sync_){
            approximate_sync_ =
                boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(1);
            approximate_sync_->connectInput(sub_nearest_roi_rect_,
                                            sub_yolo_detected_boxes_);
            approximate_sync_->registerCallback(boost::bind
                                                (&KcfTrackerROS::callback, this, _1, _2));
        } else {
            sync_  =
                boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1);
            sync_->connectInput(sub_nearest_roi_rect_,
                                sub_yolo_detected_boxes_);
            sync_->registerCallback(boost::bind
                                    (&KcfTrackerROS::callback, this, _1, _2));
        }

        frames = 0;
    }

    void KcfTrackerROS::visualize(cv::Mat& image, const BBox_c& bb, double frames)
    {
        cv::putText(image, "frame: " + std::to_string(frames),
                    cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "kernel_sigma: " + std::to_string(kernel_sigma_),
                    cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "cell_size: " + std::to_string(cell_size_),
                    cv::Point(50, 110), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "num_scales: " + std::to_string(num_scales_),
                    cv::Point(50, 140), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
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

    void KcfTrackerROS::publish_messages(const cv::Mat& image, const cv::Mat& croped_image,
                                         const BBox_c& bb, bool changed)
    {
        kcf_ros::Rect output_rect;
        output_rect.x = bb.cx - bb.w * 0.5; // left top x
        output_rect.y = bb.cy - bb.h * 0.5; // left top y
        output_rect.width = bb.w;
        output_rect.height = bb.h;
        output_rect.changed = changed;
        output_rect.header = header_;
        output_rect_pub_.publish(output_rect);
        debug_image_pub_.publish(cv_bridge::CvImage(header_,
                                                    sensor_msgs::image_encodings::BGR8,
                                                    image).toImageMsg());
        croped_image_pub_.publish(cv_bridge::CvImage(header_,
                                                     sensor_msgs::image_encodings::BGR8,
                                                     croped_image).toImageMsg());
    }

    bool KcfTrackerROS::boxesToBox(const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes,
                                   const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg,
                                   cv::Rect& output_box)
    {
        // TODO: choose a box from time-series data of detected boxes,
        // NOW: closest box from croped roi image center

        cv::Point2f nearest_roi_image_center(nearest_roi_rect_msg->width,
                                             nearest_roi_rect_msg->height);

        float min_distance = std::pow(24, 24);
        cv::Rect box_on_nearest_roi_image;
        for (auto box : detected_boxes->objects) {

            float center_to_detected_box_distance =
                cv::norm(cv::Point2f(box.x + box.width * 0.5, box.y + box.height * 0.5) - nearest_roi_image_center);
            if (center_to_detected_box_distance < min_distance) {
                box_on_nearest_roi_image = cv::Rect(box.x, box.y, box.width, box.height);
                min_distance = center_to_detected_box_distance;
            }
        }
        output_box = box_on_nearest_roi_image;
        return true;
    }

    float KcfTrackerROS::check_detecter_confidence(const std::vector<cv::Rect> detecter_results)
    {
        float confidence = 0;

        std::cerr << "detecter_results.size(): " << detecter_results.size() << std::endl;

        auto current_result = detecter_results.at(detecter_results.size() - 1);
        auto prev_result = detecter_results.at(detecter_results.size() - 2);

        cv::Point2f current_result_center(current_result.x + current_result.width * 0.5,
                                          current_result.y + current_result.height * 0.5);

        cv::Point2f prev_result_center(prev_result.x + prev_result.width * 0.5,
                                       prev_result.y + prev_result.height * 0.5);

        float distance =
            cv::norm(current_result_center - prev_result_center);
        if (distance < detecter_threshold_) {
            confidence = 1;
        } else {
            confidence = 0;
        }
        return confidence;
    }

    float KcfTrackerROS::check_tracker_confidence(const std::vector<BBox_c> tracker_results)
    {
        float confidence = 0;

        std::cerr << "tracker_results.size(): " << tracker_results.size() << std::endl;

        auto current_result = tracker_results.at(tracker_results.size() - 1);
        auto prev_result = tracker_results.at(tracker_results.size() - 2);

        float distance =
            cv::norm(cv::Point2f(current_result.cx, current_result.cy) - cv::Point2f(prev_result.cx, prev_result.cy));
        if (distance < tracker_threshold_) {
            confidence = 1;
        } else {
            confidence = 0;
        }
        return confidence;
    }

    void KcfTrackerROS::image_callback(const sensor_msgs::Image::ConstPtr& raw_image_msg)
    {
        boost::mutex::scoped_lock lock(mutex_);

        std::cerr << "---------" << __func__ << std::endl;
        std::cerr << "raw_image_msg->header.stamp: " << raw_image_msg->header.stamp << std::endl;

        header_ = raw_image_msg->header;
        image_.release();
        load_image(image_, raw_image_msg);

        if (!tracker_initialized_){
            if (debug_log_)
                ROS_INFO("wait for tracker initialized");
            return;
        }

        if (track_flag_) {
            double time_profile_counter = cv::getCPUTickCount();
            tracker.track(image_);
            time_profile_counter = cv::getCPUTickCount() - time_profile_counter;

            if (debug_log_) {
                float time = time_profile_counter / ((double)cvGetTickFrequency() * 1000);
                ROS_INFO("frame%d: speed-> %f ms/frame", frames, time);
            }

            BBox_c bb = tracker.getBBox();

            cv::Point lt(bb.cx - bb.w * 0.5, bb.cy - bb.h * 0.5);
            cv::Point rb(bb.cx + bb.w * 0.5, bb.cy + bb.h * 0.5);
            if (rb.x > image_.cols) rb.x = image_.cols;
            if (rb.y > image_.rows) rb.y = image_.rows;
            if (lt.x < 0)  lt.x = 0;
            if (lt.y < 0) lt.y = 0;
            int width = rb.x - lt.x;
            int height = rb.y - lt.y;

            // ROS_INFO("lt.x, lt.y, rb.x, rb.y, width, height: %d, %d, %d, %d, %d, %d",
            //          lt.x, lt.y, rb.x, rb.y, width, height);

            if (tracker_results_queue_.size() > queue_size_) {
                tracker_results_queue_.erase(tracker_results_queue_.begin());
            }
            tracker_results_queue_.push_back(bb);

            float confidence = 1;
            if (detecter_results_queue_.size() > 5) {
                confidence = check_tracker_confidence(tracker_results_queue_);
            }

            if (confidence > 0.5) {
                ROS_INFO("confidence: %f", confidence);
                if (width > 0.0 && height > 0.0){
                    std::cerr << width << ", " << height << std::endl;
                    cv::Mat croped_image =
                        image_(cv::Rect(lt.x, lt.y, width, height)).clone();
                    visualize(image_, bb, frames);
                    publish_messages(image_, croped_image, bb, signal_changed_);
                }
            } else {
                ROS_INFO("confidence: %f, track_flag_ = false", confidence);
                track_flag_ = false;
            }
        }
        frames++;
    }

    void KcfTrackerROS::callback(const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg,
                                 const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes)
    {
        boost::mutex::scoped_lock lock(mutex_);

        std::cerr << "---------" << __func__ << std::endl;
        std::cerr << "detected_boxes->header.stamp: " << detected_boxes->header.stamp << std::endl;

        // if (image_.empty() || non_detected_count_ > 3) {
        //   if (debug_log_)
        //     ROS_INFO("wait for first image or detected box");
        //   return;
        // }

        if (image_.empty()) {
            if (debug_log_)
                ROS_INFO("wait for first image");
            return;
        }

        if (non_detected_count_ > 5) {
            track_flag_ = false;
        }

        if (prev_signal_ != nearest_roi_rect_msg->signal){
            signal_changed_ = true;
        } else {
            signal_changed_ = false;
        }

        // TODO: detect outlier of tracking result by time-series data processing
        // if (signal_changed_ || callback_count_ == 0){
        if (detected_boxes->objects.size() > 0){
            if (debug_log_)
                ROS_WARN("init box on raw image!!!");

            non_detected_count_ = 0;
            track_flag_ = true;

            cv::Rect box_on_nearest_roi_image;
            boxesToBox(detected_boxes, nearest_roi_rect_msg, box_on_nearest_roi_image);

            // Yolo output box is detected on croped roi image by feat_proj,
            // nearest_roi_rect_msg has a location of croped roi image on original raw image
            cv::Rect init_box_on_raw_image(box_on_nearest_roi_image.x + nearest_roi_rect_msg->x - offset_,
                                           box_on_nearest_roi_image.y + nearest_roi_rect_msg->y - offset_,
                                           box_on_nearest_roi_image.width + offset_ * 2,
                                           box_on_nearest_roi_image.height + offset_ * 2);


            if (detecter_results_queue_.size() > 5) {
                std::cerr << "erase" << std::endl;
                detecter_results_queue_.erase(detecter_results_queue_.begin());
            }
            std::cerr << "push_back" << std::endl;
            detecter_results_queue_.push_back(init_box_on_raw_image);

            float confidence = 1;
            if (detecter_results_queue_.size() > 5) {
                confidence = check_detecter_confidence(detecter_results_queue_);
            }

            if (confidence > 0.5) {
                ROS_INFO("confidence: %f", confidence);
                tracker.init(image_, init_box_on_raw_image);
                tracker_initialized_ = true;
            } else {
                ROS_INFO("confidence: %f", confidence);
            }

        } else {
            non_detected_count_++;
        }

        callback_count_++;
        prev_signal_ = nearest_roi_rect_msg->signal;
    }
} // namespace kcf_ros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(kcf_ros::KcfTrackerROS, nodelet::Nodelet)
