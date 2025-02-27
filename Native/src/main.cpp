#include <iostream>
#include "inference_engine.h"
#include "image_processor.h"

// 添加必要的 OpenCV 頭文件
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>  // 用於 imshow, waitKey
#include <opencv2/imgcodecs.hpp> // 用於 imwrite
#include <opencv2/imgproc.hpp>   // 用於 addWeighted
#include <chrono>  // 添加計時功能

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }
    
    try {
        // 創建適合當前平台的推理引擎
        auto engine = InferenceEngine::create(argv[1]);
        
        // 載入並預處理圖像
        cv::Mat image = ImageProcessor::loadAndPreprocess(argv[2]);
        
        // 執行推理並獲取結果
        cv::Mat result = engine->predict(image);
        
        cv::imwrite("result.png", result);
        // 性能測試：重複推理 100 次
        const int num_tests = 0;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_tests; ++i) {
            result = engine->predict(image.clone());

            // cv::imwrite("result"+std::to_string(i)+".png", result);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        cv::imwrite("input.png", image);
        // 計算 FPS
        double fps = 1000.0 * num_tests / duration.count();
        std::cout << "Performance test results:" << std::endl;
        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average time per frame: " << duration.count() / num_tests << " ms" << std::endl;
        std::cout << "FPS: " << fps << std::endl;
        
        // 顯示並保存結果
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