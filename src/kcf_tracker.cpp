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
        pnh_.getParam("offset", offset_);

        if (debug_log_) {
            std::cout << "kernel_sigma: " << kernel_sigma_ << std::endl;
            std::cout << "cell_size: " << cell_size_ << std::endl;
            std::cout << "num_scales: " << num_scales_ << std::endl;
        }

        if (debug_view_) {
            // cv::namedWindow("output", CV_WINDOW_NORMAL);
            // cv::namedWindow("interpolation", CV_WINDOW_NORMAL);
            // cv::namedWindow("debug", CV_WINDOW_NORMAL);
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

        frames = 0;
    }

    void KcfTrackerROS::visualize(cv::Mat& image,
                                  const cv::Rect& rect,
                                  const cv::Rect& nearest_roi_rect,
                                  double frames,
                                  float box_movement_ratio)
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

        cv::rectangle(image, rect, CV_RGB(0,255,0), 2);

        cv::rectangle(image, cv::Rect(nearest_roi_rect.x, nearest_roi_rect.y,
                                      nearest_roi_rect.width, nearest_roi_rect.height),
                      CV_RGB(255,0,0), 2);

        if (debug_view_){
            // cv::namedWindow("output", CV_WINDOW_NORMAL);
            cv::imshow("output", image);
            cv::waitKey(10);
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
            std::cerr << "tmp1: " << tmp1 << std::endl;

            auto v_t = (input_vec - distribution.mean).transpose();
            std::cout << "v_t: " << v_t << std::endl;

            auto v = (distribution.cov_inverse * (input_vec - distribution.mean));
            std::cout << "v: " << v << std::endl;

            double tmp2 = std::exp(-0.5 * v_t * v);
            std::cerr << "tmp2: " << tmp2 << std::endl;

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
        // ROS_WARN("score, w_distance, w_score: %f, %f, %f", score, w_distance, w_score);

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

        // std::cerr << "---" << std::endl;
        // std::cerr << detected_boxes->objects.size() << std::endl;

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
            // ROS_WARN("no traffic light in detection results");
            return false;
        }

        return true;
    }

    float KcfTrackerROS::check_detector_confidence(const std::vector<cv::Rect> detector_results,
                                                   const float detection_score,
                                                   float& movement,
                                                   cv::Rect& init_box_on_raw_image)
    {
        float confidence = 0;

        auto current_result = detector_results.at(detector_results.size() - 1);
        auto prev_result = detector_results.at(detector_results.size() - 2);

        cv::Point2f current_result_center(current_result.x + current_result.width * 0.5,
                                          current_result.y + current_result.height * 0.5);
        cv::Point2f prev_result_center(prev_result.x + prev_result.width * 0.5,
                                       prev_result.y + prev_result.height * 0.5);

        float distance =
            cv::norm(current_result_center - prev_result_center);
        // ROS_WARN("detector distance: %f", distance);

        float raw_image_diagonal = Eigen::Vector2d(raw_image_width_, raw_image_height_).norm();
        float box_movement_ratio = distance / raw_image_diagonal;
        ROS_WARN("box_movement_ratio: %f", box_movement_ratio);

        // ROS_WARN("prev_result.height, prev_result.width: %d, %d", prev_result.height, prev_result.width);

        float prev_box_ratio = float(prev_result.height) / float(prev_result.width);
        float current_box_ratio = float(current_result.height) / float(current_result.width);

        float box_size_confidence;
        int largest_ratio_index = 0; // prev:-2 current: -1
        if (float(prev_box_ratio) > float(current_box_ratio)) {
            box_size_confidence = float(current_box_ratio) / float(prev_box_ratio);
            largest_ratio_index = -2;
        } else {
            box_size_confidence = float(prev_box_ratio)/ float(current_box_ratio);
            largest_ratio_index = -1;
        }

        ROS_WARN("box_size_confidence, current_box_ratio, prev_box_ratio: %f, %f, %f",
                 box_size_confidence, current_box_ratio, prev_box_ratio);

        // box size significantly changed even box position almost unchanged between prev frame and current frame
        if (box_movement_ratio < 0.05 and box_size_confidence < 0.7) {
            // ROS_WARN("box_movement_ratio < 0.05 and box_size_confidence < 0.7: do not update box size");
            // if (largest_ratio_index == -2) {
            //     init_box_on_raw_image.width = prev_result.width;
            //     init_box_on_raw_image.height = prev_result.height;
            // } else if (largest_ratio_index == -1) {
            //     init_box_on_raw_image.width = current_result.width;
            //     init_box_on_raw_image.height = current_result.height;
            // } else {
            //     ROS_ERROR("error on %s", __func__);
            // }

            ROS_WARN("box_movement_ratio < 0.05 and box_size_confidence < 0.7: confidence 0");
            confidence = 0;
            return confidence;
        }

        ROS_WARN("box_movement_ratio > 0.05 and box_size_confidence > 0.7");

        float detector_threshold;
        if (detection_score > 0.8) {
            detector_threshold = 0.5;
        } else {
            detector_threshold = 0.05;
        }

        if (box_movement_ratio < detector_threshold) {
            confidence = 1;
        } else {
            confidence = 0;
        }

        movement = box_movement_ratio;
        return confidence;
    }

    float KcfTrackerROS::check_tracker_confidence(const std::vector<BBox_c> tracker_results)
    {
        float confidence = 0;

        auto current_result = tracker_results.at(tracker_results.size() - 1);
        auto prev_result = tracker_results.at(tracker_results.size() - 2);

        float distance =
            cv::norm(cv::Point2f(current_result.cx, current_result.cy) - cv::Point2f(prev_result.cx, prev_result.cy));
        ROS_INFO("check tracker: distance: %f", distance);
        if (distance < tracker_threshold_) {
            confidence = 1;
        } else {
            confidence = 0;
        }
        return confidence;
    }

    bool KcfTrackerROS::enqueue_detection_results(const cv::Rect& init_box_on_raw_image)
    {
        try {
            if (detector_results_queue_.size() >= queue_size_)
                detector_results_queue_.erase(detector_results_queue_.begin());
            detector_results_queue_.push_back(init_box_on_raw_image);
            return true;
        } catch (...) {
            std::cerr << "exception ..." << std::endl;
            return false;
        }
    }

    bool KcfTrackerROS::enqueue_tracking_results(const BBox_c& bb)
    {
        try {
            if (tracker_results_queue_.size() >= queue_size_)
                tracker_results_queue_.erase(tracker_results_queue_.begin());
            tracker_results_queue_.push_back(bb);
            return true;
        } catch (...) {
            std::cerr << "exception ..." << std::endl;
            return false;
        }
    }

    void KcfTrackerROS::boxes_callback(const autoware_msgs::DetectedObjectArray::ConstPtr& detected_boxes){
        boost::mutex::scoped_lock lock(mutex_);
        detected_boxes_ = detected_boxes;
        detected_boxes_stamp_ = detected_boxes_->header.stamp.toSec();
        boxes_callback_cnt_++;
    }

    int KcfTrackerROS::get_min_index(){
        int min_index = 0;
        for (int i=image_stamps.size() - 1; i>0; i--) {
            if (image_stamps.at(i) - detected_boxes_stamp_ < 0) {
                if (std::abs(image_stamps.at(i+1) - detected_boxes_stamp_) <
                    std::abs(image_stamps.at(i) - detected_boxes_stamp_)){
                    min_index = i+1;
                } else {
                    min_index = i;
                }
                break;
            }
        }
        return min_index;
    }

    bool KcfTrackerROS::update_tracker(cv::Mat& image, cv::Rect& output_rect) {
        try {
            double start_time = ros::Time::now().toSec();
            tracker.track(image);
            double total_time = ros::Time::now().toSec() - start_time;
            if (debug_log_)
                ROS_INFO("interpolation total time: %.2lf [ms]",
                         total_time * 1000);

            // double time_profile_counter = cv::getCPUTickCount();
            // tracker.track(image);
            // time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
            // if (debug_log_)
            //     ROS_INFO("speed: %.2f [ms]", time_profile_counter/((double)cvGetTickFrequency()*1000));

            BBox_c bb = tracker.getBBox();
            // ROS_WARN("bb: (%f, %f, %f, %f)", bb.cx, bb.cy, bb.w, bb.h);

            cv::Point lt(bb.cx - bb.w * 0.5, bb.cy - bb.h * 0.5);
            cv::Point rb(bb.cx + bb.w * 0.5, bb.cy + bb.h * 0.5);
            if (rb.x > image.cols) rb.x = image.cols;
            if (rb.y > image.rows) rb.y = image.rows;
            if (lt.x < 0)  lt.x = 0;
            if (lt.y < 0) lt.y = 0;
            int width = rb.x - lt.x;
            int height = rb.y - lt.y;
            output_rect = cv::Rect(lt.x, lt.y, width, height);
        } catch (...) {
            ROS_ERROR("failed tracker update ");
            return false;
        }

        return true;
    }

    bool KcfTrackerROS::box_interpolation(int min_index){
        ROS_INFO("buffer_size: %d", image_buffer.size() - min_index - 1);

        for (int i=min_index; i<image_buffer.size(); i++) {

            if (i == min_index) {
                cv::Mat debug_image = image_buffer.at(i).clone();
                cv::Rect box_on_nearest_roi_image;
                float detection_score;
                if (boxesToBox(detected_boxes_, rect_buffer.at(i), box_on_nearest_roi_image, detection_score)) {
                    cv::Rect init_box_on_raw_image(box_on_nearest_roi_image.x + rect_buffer.at(i).x - offset_,
                                                   box_on_nearest_roi_image.y + rect_buffer.at(i).y - offset_,
                                                   box_on_nearest_roi_image.width + offset_ * 2,
                                                   box_on_nearest_roi_image.height + offset_ * 2);
                    if (debug_view_) {
                        cv::rectangle(debug_image, init_box_on_raw_image, CV_RGB(0,0,255), 4);
                        cv::rectangle(debug_image,
                                      cv::Rect(rect_buffer.at(i).x, rect_buffer.at(i).y,
                                               rect_buffer.at(i).width, rect_buffer.at(i).height),
                                      CV_RGB(255,0,0), 2);

                        cv::imshow("output", debug_image);
                        cv::waitKey(10);
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
                if (!update_tracker(image_buffer.at(i), output_rect))
                    return false;
                // visualize(image_buffer.at(i), output_rect, rect_buffer.at(i), frames);
            }
        }
        return true;
    }

    bool KcfTrackerROS::create_buffer(const cv::Mat &image,
                                      double image_stamp,
                                      const cv::Rect &rect){
        if (image_stamps.size() >= buffer_size_) {
            image_stamps.erase(image_stamps.begin());
            image_buffer.erase(image_buffer.begin());
            rect_buffer.erase(rect_buffer.begin());
        }
        image_stamps.push_back(image_stamp);
        image_buffer.push_back(image);
        rect_buffer.push_back(rect);
        return true;
    }

    void KcfTrackerROS::callback(const sensor_msgs::Image::ConstPtr& raw_image_msg,
                                 const kcf_ros::Rect::ConstPtr& nearest_roi_rect_msg)
    {
        cv::namedWindow("output", CV_WINDOW_NORMAL);
        boost::mutex::scoped_lock lock(mutex_);
        std::cerr << "------------------" << __func__ << std::endl;
        std::cerr << "frame: " << frames << std::endl;
        cv::Mat image;
        load_image(image, raw_image_msg);
        create_buffer(image,
                      raw_image_msg->header.stamp.toSec(),
                      cv::Rect(nearest_roi_rect_msg->x,
                               nearest_roi_rect_msg->y,
                               nearest_roi_rect_msg->width,
                               nearest_roi_rect_msg->height));

        float nearest_stamp = 0;
        if (boxes_callback_cnt_ != prev_boxes_callback_cnt_) {
            int min_index = get_min_index();;

            ROS_INFO("start box interpolate");

            double start_time = ros::Time::now().toSec();
            bool success_interpolation = box_interpolation(min_index);
            double total_time = ros::Time::now().toSec() - start_time;
            if (debug_log_)
                ROS_INFO("interpolation total time: %.2lf [ms]",
                         total_time * 1000);

        } else if (prev_boxes_callback_cnt_ == 0) {
        } else {
            ROS_INFO("default tracking");
            cv::Rect output_rect;
            update_tracker(image, output_rect);
            visualize(image,
                      output_rect,
                      cv::Rect(nearest_roi_rect_msg->x,
                               nearest_roi_rect_msg->y,
                               nearest_roi_rect_msg->width,
                               nearest_roi_rect_msg->height),
                      frames);
        }

        prev_signal_ = nearest_roi_rect_msg->signal;
        prev_boxes_callback_cnt_ = boxes_callback_cnt_;
        frames++;
    }
} // namespace kcf_ros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(kcf_ros::KcfTrackerROS, nodelet::Nodelet)
