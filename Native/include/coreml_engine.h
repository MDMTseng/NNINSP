#pragma once
#include "inference_engine.h"
#include <vector>

// Forward declaration for OpenCV
namespace cv {
    class Mat;
}

// C++ wrapper that implements InferenceEngine interface
class CoreMLEngine : public InferenceEngine {
private:
    void* objc_engine_;  // Pointer to Objective-C engine
    void* objc_output_;  // Pointer to current MLMultiArray output
    ModelInfo model_info_;
    std::vector<std::string> _class_names;
    
public:
    CoreMLEngine(const std::string& model_path);
    ~CoreMLEngine();
    
    cv::Mat predict(const cv::Mat& image) override;
    cv::Mat predict_batch(const std::vector<cv::Mat>& images);
    const ModelInfo& getModelInfo() const override;
    std::vector<std::string> get_class_names() const;
    
private:
    std::string findModelInfoPath(const std::string& model_path);
    cv::Mat preprocess(const cv::Mat& input);
}; 