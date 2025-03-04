#include "model_info.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

ModelInfo::ModelInfo(const std::string& info_path) {
    std::ifstream file(info_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model info file: " + info_path);
    }
    
    std::string line;
    std::unordered_map<std::string, std::string> values;
    
    // Read key-value pairs from file
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse key=value
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            values[key] = value;
        }
    }
    
    // Extract values
    model_name = values["MODEL_NAME"];
    model_variant = values["MODEL_VARIANT"];
    num_classes = std::stoi(values["NUM_CLASSES"]);
    class_names = splitString(values["CLASS_NAMES"], ',');
    input_shape = parseIntArray(values["INPUT_SHAPE"]);
    output_shape = parseIntArray(values["OUTPUT_SHAPE"]);
    input_scale = std::stof(values["INPUT_SCALE"]);
    input_bias = parseFloatArray(values["INPUT_BIAS"]);
    color_layout = values["COLOR_LAYOUT"];
    creation_date = values["CREATION_DATE"];
    
    // Validate
    if (class_names.size() != static_cast<size_t>(num_classes)) {
        std::cerr << "Warning: NUM_CLASSES (" << num_classes << ") doesn't match CLASS_NAMES count (" 
                  << class_names.size() << ")" << std::endl;
    }
    
    std::cout << "Loaded model info: " << model_variant << " with " << num_classes 
              << " classes" << std::endl;
}

std::string ModelInfo::getClassName(int idx) const {
    if (idx >= 0 && idx < static_cast<int>(class_names.size())) {
        return class_names[idx];
    }
    return "Unknown";
}

std::vector<std::string> ModelInfo::splitString(const std::string& str, char delimiter) const {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<int> ModelInfo::parseIntArray(const std::string& str) const {
    std::vector<std::string> tokens = splitString(str, ',');
    std::vector<int> result;
    for (const auto& token : tokens) {
        result.push_back(std::stoi(token));
    }
    return result;
}

std::vector<float> ModelInfo::parseFloatArray(const std::string& str) const {
    std::vector<std::string> tokens = splitString(str, ',');
    std::vector<float> result;
    for (const auto& token : tokens) {
        result.push_back(std::stof(token));
    }
    return result;
} 