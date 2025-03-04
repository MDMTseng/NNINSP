#include <iostream>
#include "inference_engine.h"
#include "image_processor.h"

// 添加必要的 OpenCV 頭文件
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>  // 用於 imshow, waitKey
#include <opencv2/imgcodecs.hpp> // 用於 imwrite
#include <opencv2/imgproc.hpp>   // 用於 addWeighted
#include <chrono>  // 添加計時功能

// Forward declaration of helper function
cv::Mat createColoredResult(const cv::Mat& prediction, const ModelInfo& model_info);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }
    
    try {
        // Create inference engine
        auto engine = InferenceEngine::create(argv[1]);
        
        // Get model info
        const ModelInfo& model_info = engine->getModelInfo();
        std::cout << "Model: " << model_info.getModelVariant() << std::endl;
        std::cout << "Classes: ";
        for (const auto& class_name : model_info.getClassNames()) {
            std::cout << class_name << " ";
        }
        std::cout << std::endl;
        
        // Get input shape from model info
        auto input_shape = model_info.getInputShape();
        int target_size = input_shape[2]; // Height dimension
        
        // Load and preprocess image
        cv::Mat image = ImageProcessor::loadAndPreprocess(argv[2], target_size);
        
        // Run inference and get result
        cv::Mat result = engine->predict(image);
        
        // Performance test
        const int num_tests = 50;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_tests; ++i) {
            result = engine->predict(image);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Save input image
        cv::imwrite("input.png", image);
        
        // Calculate FPS
        double fps = 1000.0 * num_tests / duration.count();
        std::cout << "Performance test results:" << std::endl;
        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average time per frame: " << duration.count() / num_tests << " ms" << std::endl;
        std::cout << "FPS: " << fps << std::endl;
        

        std::cout << "Result shape: " << result.rows << "x" << result.cols << "x" << result.channels() << std::endl;
        std::cout << "Pixel format: " << result.type() << std::endl;


        // Create visualization with class colors
        cv::imwrite("_result.png", result);
        // Display and save results
        cv::Mat display;
        cv::addWeighted(image, 0.7, result, 0.3, 0, display);
        
        cv::imshow("Result", display);
        cv::imwrite("display.png", display);
        cv::waitKey(0);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// Helper function to create colored visualization
cv::Mat createColoredResult(const cv::Mat& prediction, const ModelInfo& model_info) {
    cv::Mat colored(prediction.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Create color map for classes
    std::vector<cv::Vec3b> colors = {
        cv::Vec3b(0, 0, 0),       // Background - black
        cv::Vec3b(255, 0, 0),     // Class 1 - red
        cv::Vec3b(0, 255, 0),     // Class 2 - green
        cv::Vec3b(0, 0, 255),     // Class 3 - blue
        cv::Vec3b(255, 255, 0),   // Class 4 - yellow
        cv::Vec3b(255, 0, 255),   // Class 5 - magenta
        cv::Vec3b(0, 255, 255),   // Class 6 - cyan
    };
    
    // Add more colors if needed
    while (colors.size() < static_cast<size_t>(model_info.getNumClasses())) {
        colors.push_back(cv::Vec3b(rand() % 255, rand() % 255, rand() % 255));
    }
    
    // Apply colors based on class index
    for (int y = 0; y < prediction.rows; y++) {
        for (int x = 0; x < prediction.cols; x++) {
            int class_idx = prediction.at<uchar>(y, x);
            if (class_idx < model_info.getNumClasses()) {
                colored.at<cv::Vec3b>(y, x) = colors[class_idx];
            }
        }
    }
    
    return colored;
} 