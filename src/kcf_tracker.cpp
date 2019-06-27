#include <ros/ros.h>
#include <stdlib.h>

#include "kcf.h"
#include "vot.hpp"

int main(int argc, char** argv)
{
    std::string dir_name;
    std::string kernel_sigma_str;
    std::string cell_size_str;
    std::string num_scales_str;

    if (argc == 1){
        dir_name = "./";
        kernel_sigma_str = "0.5";
        cell_size_str = "4";
        num_scales_str = "7";
    } else {
        dir_name = argv[1];
        kernel_sigma_str = argv[2];
        cell_size_str = argv[3];
        num_scales_str = argv[4];
    }

    std::cout << "dir_name: " << dir_name << std::endl;
    dir_name = dir_name + "/";

    double kernel_sigma = std::stod(std::string(kernel_sigma_str));
    double cell_size = std::stod(std::string(cell_size_str));
    double num_scales = std::stod(std::string(num_scales_str));
    std::cout << "kernel_sigma: " << kernel_sigma << std::endl;
    std::cout << "cell_size: " << cell_size << std::endl;
    std::cout << "num_scales: " << num_scales << std::endl;

    //load region, images and prepare for output
    VOT vot_io(dir_name + "region.txt", dir_name + "images.txt", dir_name + "output.txt");


    KCF_Tracker tracker(1.5, kernel_sigma, 1e-4, 0.02, 0.1, cell_size, num_scales);
    cv::Mat image;

    //img = firts frame, initPos = initial position in the first frame
    cv::Rect init_rect = vot_io.getInitRectangle();
    vot_io.outputBoundingBox(init_rect);
    vot_io.getNextImage(image);
    tracker.init(image, init_rect);
    BBox_c bb;
    double avg_time = 0.;
    int frames = 0;
    while (vot_io.getNextImage(image) == 1){
        double time_profile_counter = cv::getCPUTickCount();
        tracker.track(image);
        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
        std::cout << "frame: " << frames << std::endl;
        std::cout << "  -> speed : " <<  time_profile_counter/((double)cvGetTickFrequency()*1000) << "ms. per frame" << std::endl;
        avg_time += time_profile_counter/((double)cvGetTickFrequency()*1000);
        frames++;

        bb = tracker.getBBox();
        vot_io.outputBoundingBox(cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h));

        cv::putText(image, "frame: " + std::to_string(frames),
                    cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::putText(image, "kernel_sigma: " + kernel_sigma_str,
                    cv::Point(100, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::putText(image, "cell_size: " + cell_size_str,
                    cv::Point(100, 300), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::putText(image, "num_scales: " + num_scales_str,
                    cv::Point(100, 400), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, CV_AA);
        cv::rectangle(image, cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h), CV_RGB(0,255,0), 2);
        cv::namedWindow("output", CV_WINDOW_NORMAL);
        cv::imshow("output", image);
        cv::waitKey();
    }

    std::cout << "Average processing speed " << avg_time/frames <<  "ms. (" << 1./(avg_time/frames)*1000 << " fps)" << std::endl;

    return EXIT_SUCCESS;
}
