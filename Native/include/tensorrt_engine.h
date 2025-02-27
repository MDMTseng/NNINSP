#pragma once
#include <string>
#include <memory>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
};

class TensorRTEngine {
public:
    TensorRTEngine(const std::string& model_path, bool force_rebuild = false);
    ~TensorRTEngine();
    
    cv::Mat predict(const cv::Mat& image);

private:
    bool buildEngine(const std::string& onnx_path);
    bool loadEngine(const std::string& engine_path);
    void saveEngine(const std::string& engine_path);
    
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    Logger logger_;
    
    // 預處理參數
    const int INPUT_H = 256;
    const int INPUT_W = 256;
    std::vector<float> mean_{0.485f, 0.456f, 0.406f};
    std::vector<float> std_{0.229f, 0.224f, 0.225f};
    
    void preprocess(const cv::Mat& input, float* gpu_input);
    cv::Mat postprocess(float* gpu_output);
}; 