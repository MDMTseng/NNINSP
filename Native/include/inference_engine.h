#pragma once
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "model_info.h"

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    
    // Factory method to create appropriate engine
    static std::unique_ptr<InferenceEngine> create(const std::string& model_path);
    
    // Run inference
    virtual cv::Mat predict(const cv::Mat& image) = 0;
    
    // Get model info
    virtual const ModelInfo& getModelInfo() const = 0;
    
    // Get class name by index
    virtual std::string getClassName(int class_index) const {
        return getModelInfo().getClassName(class_index);
    }
    
    // Get number of classes
    virtual int getNumClasses() const {
        return getModelInfo().getNumClasses();
    }
    
    // Get all class names
    virtual std::vector<std::string> getClassNames() const {
        return getModelInfo().getClassNames();
    }
}; 