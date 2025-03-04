#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class ModelInfo {
public:
    ModelInfo(const std::string& info_path);
    
    // Getters
    std::string getModelName() const { return model_name; }
    std::string getModelVariant() const { return model_variant; }
    int getNumClasses() const { return num_classes; }
    std::vector<std::string> getClassNames() const { return class_names; }
    std::vector<int> getInputShape() const { return input_shape; }
    std::vector<int> getOutputShape() const { return output_shape; }
    float getInputScale() const { return input_scale; }
    std::vector<float> getInputBias() const { return input_bias; }
    std::string getColorLayout() const { return color_layout; }
    
    // Get class name by index
    std::string getClassName(int idx) const;
    
private:
    std::string model_name;
    std::string model_variant;
    int num_classes;
    std::vector<std::string> class_names;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    float input_scale;
    std::vector<float> input_bias;
    std::string color_layout;
    std::string creation_date;
    
    // Helper methods
    std::vector<std::string> splitString(const std::string& str, char delimiter) const;
    std::vector<int> parseIntArray(const std::string& str) const;
    std::vector<float> parseFloatArray(const std::string& str) const;
}; 