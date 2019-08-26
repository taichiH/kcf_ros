#include "kcf_ros/kcf_tracker.h"
#include <time.h>

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
        pnh_.getParam("interpolation_frequency", interpolation_frequency_);
        pnh_.getParam("offset", offset_);

        if (debug_log_) {
            std::cout << "kernel_sigma: " << kernel_sigma_ << std::endl;
            std::cout << "cell_size: " << cell_size_ << std::endl;
            std::cout << "num_scales: " << num_scales_ << std::endl;
            std::cout << "interpolation frequency: " << interpolation_frequency_ << std::endl;
        }

        debug_image_pub_ = pnh_.advertise<sensor_msgs::Image>("output_image", 1);
        croped_image_pub_ = pnh_.advertise<sensor_msgs::Image>("output_croped_image", 1);
        output_rect_pub_ = pnh_.advertise<kcf_ros::Rect>("output_rect", 1);

        boxes_sub =
            pnh_.subscribe("input_yolo_detected_boxes", 1, &KcfTrackerROS::boxes_callback, this);

        sub_raw_image_.subscribe(pnh_, "input_raw_image", 1);
        sub_nearest_roi_rect_.subscribe(pnh_, "input_nearest_roi_rect", 1);

        if (is_approximate_sync_){
            approximate_sync_ =
                boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(1000);
            approximate_sync_->connectInput(sub_raw_image_,
                                            sub_nearest_roi_rect_);
            approximate_sync_->registerCallback(boost::bind
                                                (&KcfTrackerROS::callback, this, _1, _2));
        } else {
            sync_  =
                boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
            sync_->connectInput(sub_raw_image_,
                                sub_nearest_roi_rect_);
            sync_->registerCallback(boost::bind
                                    (&KcfTrackerROS::callback, this, _1, _2));
        }
    }

    void KcfTrackerROS::visualize(cv::Mat& image,
                                  const cv::Rect& rect,
                                  const cv::Rect& nearest_roi_rect,
                                  double frames,
                                  float box_movement_ratio,
                                  float tracker_conf,
                                  float tracking_time,
                                  std::string mode)
    {
        cv::putText(image, "frame: " + std::to_string(frames),
                    cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "kernel_sigma: " + std::to_string(kernel_sigma_),
                    cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "cell_size: " + std::to_string(cell_size_),
                    cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "num_scales: " + std::to_string(num_scales_),
                    cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "box_movement_ratio: " + std::to_string(box_movement_ratio),
                    cv::Point(50, 250), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "interpolation freq: " + std::to_string(interpolation_frequency_),
                    cv::Point(50, 300), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "offset: " + std::to_string(offset_),
                    cv::Point(50, 350), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "mode: " + mode,
                    cv::Point(50, 400), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "tracker_conf: " + std::to_string(tracker_conf),
                    cv::Point(50, 450), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::putText(image, "trackeing_time: " + std::to_string(tracking_time),
                    cv::Point(50, 500), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 1, CV_AA);

        if (mode == "init") {
            cv::rectangle(image, rect, CV_RGB(0,0,255), 2);
        } else {
            cv::rectangle(image, rect, CV_RGB(0,255,0), 2);
        }

        cv::rectangle(image, cv::Rect(nearest_roi_rect.x, nearest_roi_rect.y,
                                      nearest_roi_rect.width, nearest_roi_rect.height),
                      CV_RGB(255,0,0), 2);

        if (debug_view_){
            cv::imshow("output", image);
            cv::waitKey(5);
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
                                         const cv::Rect& rect, bool changed)
    {
        kcf_ros::Rect output_rect;
        output_rect.x = rect.x; // left top x
        output_rect.y = rect.y; // left top y
        output_rect.width = rect.width;
        output_rect.height = rect.height;
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

    bool KcfTrackerROS::calc_gaussian(double& likelihood,
                                      const Eigen::Vector2d& input_vec,
                                      const GaussianDistribution& distribution)
    {
        likelihood = 0;
        try {
            double tmp1 = 1 / (std::pow((2 * pi), distribution.d) * distribution.cov_det_sqrt);
            auto v_t = (input_vec - distribution.mean).transpose();
            auto v = (distribution.cov_inverse * (input_vec - distribution.mean));
            double tmp2 = std::exp(-0.5 * v_t * v);

            likelihood = tmp1 * tmp2;
            return true;
        } catch(...) {
            ROS_ERROR("error in %s", __func__);
            return false;
        }
    }


    double KcfTrackerROS::calc_detection_score(const autoware_msgs::DetectedObject& box,
                                               const cv::Point2f& nearest_roi_image_center)
    {
        double score = 0;
        double w_score = box.score;
        double w_distance = 0;

        bool gaussian_distance_ = false;
        Eigen::Vector2d input_vec(box.x + box.width * 0.5,
                                  box.y + box.height * 0.5);
        Eigen::Vector2d center_vec(nearest_roi_image_center.x,
                                   nearest_roi_image_center.y);

        double w =(nearest_roi_image_center.x * 2);
        double h =(nearest_roi_image_center.y * 2);

        // calc gaussian distance
        if (gaussian_distance_) {
            double likelihood;
            GaussianDistribution distribution;
            distribution.mean = center_vec;
            calc_gaussian(likelihood, input_vec, distribution);
            std::cerr << "likelihood: " << likelihood << std::endl;
            w_distance = likelihood;
        } else {
            double diagonal = std::sqrt(w * w + h * h);
            w_distance = 1 - (input_vec - center_vec).norm() / diagonal;
        }

        score = (w_score + w_distance) / 2;

        return score;
    }

    bool KcfTrackerROS::boxesToBox(const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes,
                                   const cv::Rect& roi_rect,
                                   cv::Rect& output_box,
                                   float& score)
    {
        // TODO: choose a box from time-series data of detected boxes,
        // NOW: closest box from croped roi image center

        if (detected_boxes->objects.size() == 0) {
            return false;
        }

        cv::Point2f nearest_roi_image_center(roi_rect.width * 0.5,
                                             roi_rect.height * 0.5);

        bool has_traffic_light = false;

        float max_score = 0;
        float min_distance = std::pow(24, 24);
        cv::Rect box_on_nearest_roi_image;

        for (autoware_msgs::DetectedObject box : detected_boxes->objects) {
            if (box.label != "traffic light")
                continue;
            has_traffic_light = true;

            // float center_to_detected_box_distance =
            //     cv::norm(cv::Point2f(box.x + box.width * 0.5, box.y + box.height * 0.5) - nearest_roi_image_center);
            // if (center_to_detected_box_distance < min_distance) {
            //     output_box = cv::Rect(box.x, box.y, box.width, box.height);
            //     score = box.score;
            //     min_distance = center_to_detected_box_distance;
            // }

            float tmp_score = calc_detection_score(box, nearest_roi_image_center);
            if (tmp_score > max_score) {
                output_box = cv::Rect(box.x, box.y, box.width, box.height);
                score = tmp_score;
                max_score = tmp_score;
            }

            // ROS_INFO("detection score: %f, box: %d, %d, %d, %d", tmp_score, box.x, box.y, box.width, box.height);
        }
        if (!has_traffic_light) {
            ROS_WARN("non traffic light label in detection results");
            return false;
        }

        return true;
    }

    float KcfTrackerROS::check_confidence(const std::vector<cv::Rect> results,
                                          float& box_movement_ratio)
    {

        box_movement_ratio = 0;

        auto current_result = results.at(results.size() - 1);
        auto prev_result = results.at(results.size() - 2);

        float distance =
            cv::norm(cv::Point2f(current_result.x + current_result.width * 0.5,
                                 current_result.y + current_result.height * 0.5) -
                     cv::Point2f(prev_result.x + prev_result.width * 0.5,
                                 prev_result.y + prev_result.height * 0.5));

        float raw_image_diagonal = Eigen::Vector2d(raw_image_width_, raw_image_height_).norm();
        box_movement_ratio = distance / raw_image_diagonal;
        return sigmoid(box_movement_ratio);
    }

    float KcfTrackerROS::sigmoid(float x, float a) {
        return(1 - (1.0 / (1.0 + exp(a * (-x + box_movement_thresh_)))));
    }

    bool KcfTrackerROS::create_tracker_results_buffer(const cv::Rect& bb)
    {
        try {
            if (tracker_results_buffer_.size() >= queue_size_)
                tracker_results_buffer_.erase(tracker_results_buffer_.begin());
            tracker_results_buffer_.push_back(bb);
            return true;
        } catch (...) {
            ROS_ERROR("failed create tracker results buffer");
            return false;
        }
    }

    bool KcfTrackerROS::get_min_index(int& min_index){
        min_index = 0;
        bool found_min_index = false;

        if (image_stamps.empty()) {
            ROS_ERROR("image_stamps is empty");
            return false;
        }

        for (int i=image_stamps.size() - 1; i>=0; i--) {
            if (image_stamps.at(i) - detected_boxes_stamp_ < 0) {
                found_min_index = true;
                if (std::abs(image_stamps.at(i+1) - detected_boxes_stamp_) <
                    std::abs(image_stamps.at(i) - detected_boxes_stamp_)){
                    min_index = i+1;
                } else {
                    min_index = i;
                }
                break;
            }
        }
        return found_min_index;
    }

    bool KcfTrackerROS::update_tracker(cv::Mat& image, cv::Rect& output_rect, const cv::Rect& roi_rect,
                                       float& box_movement_ratio, float& tracker_conf, float& tracking_time) {
        try {
            double start_time = ros::Time::now().toSec();
            tracker.track(image);
            tracking_time = ros::Time::now().toSec() - start_time;

            if (debug_log_)
                ROS_INFO("tracking time: %.2lf [ms]", tracking_time * 1000);

            BBox_c bb = tracker.getBBox();

            cv::Point lt(bb.cx - bb.w * 0.5, bb.cy - bb.h * 0.5);
            cv::Point rb(bb.cx + bb.w * 0.5, bb.cy + bb.h * 0.5);
            if (rb.x > image.cols) rb.x = image.cols;
            if (rb.y > image.rows) rb.y = image.rows;
            if (lt.x < 0)  lt.x = 0;
            if (lt.y < 0) lt.y = 0;
            int width = rb.x - lt.x;
            int height = rb.y - lt.y;
            output_rect = cv::Rect(lt.x, lt.y, width, height);

            // tracked rect is outside of roi_rect
            if (bb.cx < roi_rect.x ||
                bb.cy < roi_rect.y ||
                bb.cx > roi_rect.x + roi_rect.width ||
                bb.cy > roi_rect.y + roi_rect.height) {
                return false;
            }

            if(!create_tracker_results_buffer(output_rect))
                return false;

            box_movement_ratio = 0.0;
            tracker_conf = 0.0;
            if (tracker_results_buffer_.size() >= queue_size_) {
                tracker_conf = check_confidence(tracker_results_buffer_, box_movement_ratio);
            }

        } catch (...) {
            ROS_ERROR("failed tracker update ");
            return false;
        }

        return true;
    }


    bool KcfTrackerROS::box_interpolation(int min_index){
        if (debug_log_)
            ROS_INFO("buffer_size: %d, freq: %d, calc size: %d",
                     image_buffer.size() - min_index,
                     interpolation_frequency_,
                     int((image_buffer.size() - min_index - 1) / interpolation_frequency_));

        for (int i=min_index; i<image_buffer.size(); i+=interpolation_frequency_) {
            if (i == min_index) {
                cv::Mat debug_image = image_buffer.at(i).clone();
                cv::Rect box_on_nearest_roi_image;
                float detection_score;
                if (boxesToBox(detected_boxes_, rect_buffer.at(i), box_on_nearest_roi_image, detection_score)) {
                    cv::Rect init_box_on_raw_image(box_on_nearest_roi_image.x + rect_buffer.at(i).x - offset_,
                                                   box_on_nearest_roi_image.y + rect_buffer.at(i).y - offset_,
                                                   box_on_nearest_roi_image.width + offset_ * 2,
                                                   box_on_nearest_roi_image.height + offset_ * 2);

                    cv::Point lt(init_box_on_raw_image.x, init_box_on_raw_image.y);
                    cv::Point rb(init_box_on_raw_image.x + init_box_on_raw_image.width,
                                 init_box_on_raw_image.y + init_box_on_raw_image.height);
                    if (rb.x > image_buffer.at(i).cols) rb.x = image_buffer.at(i).cols;
                    if (rb.y > image_buffer.at(i).rows) rb.y = image_buffer.at(i).rows;
                    if (lt.x < 0)  lt.x = 0;
                    if (lt.y < 0) lt.y = 0;
                    int width = rb.x - lt.x;
                    int height = rb.y - lt.y;
                    init_box_on_raw_image = cv::Rect(lt.x, lt.y, width, height);

                    if (image_buffer.size() - min_index == 1) {
                        cv::Mat croped_image = image_buffer.at(i)(init_box_on_raw_image).clone();
                        cv::Mat vis_image = image_buffer.at(i).clone();
                        visualize(vis_image, init_box_on_raw_image, rect_buffer.at(i), image_stamps.at(i), 0, 0, 0, "init");
                        publish_messages(vis_image, croped_image, init_box_on_raw_image, signal_changed_);
                    }

                    try {
                        tracker.init(image_buffer.at(i), init_box_on_raw_image);
                    } catch (...) {
                        ROS_ERROR("failed tracker init ");
                        return false;
                    }
                } else {
                    ROS_ERROR("failed convert boxes to box");
                    return false;
                }
            } else {
                cv::Rect output_rect;
                float box_movement_ratio, tracker_conf, tracking_time;
                if (!update_tracker(image_buffer.at(i), output_rect, rect_buffer.at(i),
                                    box_movement_ratio, tracker_conf, tracking_time))
                    return false;
            }
        }
        return true;
    }

    bool KcfTrackerROS::create_buffer(const ImageInfoPtr& image_info){
        try {
            if (image_stamps.size() >= buffer_size_) {
                image_stamps.erase(image_stamps.begin());
                image_buffer.erase(image_buffer.begin());
                rect_buffer.erase(rect_buffer.begin());
            }
            image_stamps.push_back(image_info->stamp);
            image_buffer.push_back(image_info->image);
            rect_buffer.push_back(image_info->rect);
            return true;
        } catch (...) {
            ROS_ERROR("failed stack image_info buffer");
            return false;
        }
    }

    bool KcfTrackerROS::clear_buffer() {
        try {
            image_stamps.clear();
            image_buffer.clear();
            rect_buffer.clear();
            return true;
        } catch (...) {
            ROS_ERROR("failed clear image_info buffer");
            return false;
        }
    }

    void KcfTrackerROS::increment_cnt() {
        prev_signal_ = signal_;
        prev_boxes_callback_cnt_ = boxes_callback_cnt_;
        cnt_++;
    }

    int KcfTrackerROS::calc_offset(int x) {
        int offset = int((9.0 / 110.0) * x - (340.0 / 110.0));
        if (offset < 0) offset = 0;
        return offset;
    }


    void KcfTrackerROS::boxes_callback(const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes){
        boost::mutex::scoped_lock lock(mutex_);
        detected_boxes_ = detected_boxes;
        detected_boxes_stamp_ = detected_boxes_->header.stamp.toSec();
        boxes_callback_cnt_++;
    }


    void KcfTrackerROS::callback(const sensor_msgs::Image::ConstPtr& raw_image_msg,
                                 const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg)
    {
        boost::mutex::scoped_lock lock(mutex_);
        std::cerr << "------------------" << __func__ << std::endl;

        if (cnt_ == 0 and debug_view_) {
            cv::namedWindow("output", CV_WINDOW_NORMAL);
            cv::resizeWindow("output", 1400, 1000);
        }

        cv::Mat image;
        load_image(image, raw_image_msg);
        raw_image_width_, raw_image_height_ = image.cols, image.rows;

        ImageInfoPtr image_info(new ImageInfo(image,
                                              cv::Rect(nearest_roi_rect_msg->x,
                                                       nearest_roi_rect_msg->y,
                                                       nearest_roi_rect_msg->width,
                                                       nearest_roi_rect_msg->height),
                                              nearest_roi_rect_msg->signal,
                                              raw_image_msg->header.stamp.toSec()));
        signal_ = nearest_roi_rect_msg->signal;
        signal_changed_ = signal_ != prev_signal_;
        offset_ = calc_offset(nearest_roi_rect_msg->height);

        if (signal_changed_) {
            clear_buffer();
            increment_cnt();
            track_flag_ = false;
            tracker_initialized_ = false;
            return;
        } else {
            create_buffer(image_info);
            track_flag_ = true;
        }

        float nearest_stamp = 0;
        if (boxes_callback_cnt_ != prev_boxes_callback_cnt_) {
            double start_time = ros::Time::now().toSec();
            int min_index = 0;
            if (!get_min_index(min_index)) {
                ROS_WARN("cannot find correspond index ...");
                track_flag_ = false;
                increment_cnt();
                return;
            }

            if (!box_interpolation(min_index)) {
                increment_cnt();
                return;
            }
            tracker_initialized_ = true;

            double total_time = ros::Time::now().toSec() - start_time;
            if (debug_log_)
                ROS_INFO("interpolation total time: %.2lf [ms]", total_time * 1000);
        } else if (prev_boxes_callback_cnt_ == 0) {
            tracker_initialized_ = false;
            track_flag_ = false;
            increment_cnt();
            return;
        } else {
            cv::Rect output_rect;
            float box_movement_ratio, tracker_conf, tracking_time;
            if (track_flag_ && tracker_initialized_) {

                if (!update_tracker(image, output_rect, image_info->rect,
                                    box_movement_ratio, tracker_conf, tracking_time)) {
                    increment_cnt();
                    track_flag_ = false;
                    tracker_initialized_ = false;
                    return;
                }
                if (tracker_conf < 0.8) {
                    increment_cnt();
                    track_flag_ = false;
                    tracker_initialized_ = false;
                    return;
                }

            } else {
                increment_cnt();
                return;
            }

            cv::Mat croped_image = image(output_rect).clone();
            cv::Mat vis_image = image.clone();
            visualize(vis_image, output_rect, image_info->rect, raw_image_msg->header.stamp.toSec(),
                      box_movement_ratio, tracker_conf, tracking_time*1000, "default");
            publish_messages(vis_image, croped_image, output_rect, signal_changed_);
        }
        increment_cnt();
    }
} // namespace kcf_ros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(kcf_ros::KcfTrackerROS, nodelet::Nodelet)
