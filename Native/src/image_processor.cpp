#include "image_processor.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat ImageProcessor::loadAndPreprocess(const std::string& image_path, int target_size) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_size, target_size));
    return resized;
}