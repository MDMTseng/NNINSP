#pragma once
#include <string>

// 前向聲明
namespace cv {
    class Mat;
}

class ImageProcessor {
public:
    static cv::Mat loadAndPreprocess(const std::string& image_path, int target_size = 256);
}; 