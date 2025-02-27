#pragma once
#include <memory>
#include <string>

// 前向聲明
namespace cv {
    class Mat;
}

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual cv::Mat predict(const cv::Mat& image) = 0;
    
    static std::unique_ptr<InferenceEngine> create(const std::string& model_path);
}; 