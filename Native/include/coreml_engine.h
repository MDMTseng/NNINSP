#pragma once
#include "inference_engine.h"

// 前向聲明，避免在頭文件中包含 OpenCV
namespace cv {
    class Mat;
}

class CoreMLEngine : public InferenceEngine {
public:
    CoreMLEngine(const std::string& model_path);
    ~CoreMLEngine();
    cv::Mat predict(const cv::Mat& image) override;

private:
    void* model_;  // MLModel*，使用 void* 避免在頭文件中包含 Objective-C
    cv::Mat preprocess(const cv::Mat& input);
    cv::Mat postprocess(const float* output, size_t size);
}; 