#include "tensorrt_engine.h"
#include <fstream>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << msg << std::endl;
    }
}

TensorRTEngine::TensorRTEngine(const std::string& model_path, bool force_rebuild) {
    std::string engine_path = model_path + ".engine";
    
    if (!force_rebuild && std::ifstream(engine_path).good()) {
        if (loadEngine(engine_path)) {
            return;
        }
    }
    
    if (!buildEngine(model_path)) {
        throw std::runtime_error("Failed to build TensorRT engine");
    }
    saveEngine(engine_path);
}

TensorRTEngine::~TensorRTEngine() {
    // RAII cleanup handled by unique_ptr
}

bool TensorRTEngine::buildEngine(const std::string& onnx_path) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

    const auto explicit_batch = 
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch));
    if (!network) return false;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) return false;

    // 解析 ONNX
    if (!parser->parseFromFile(onnx_path.c_str(), 
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        return false;
    }

    // 配置優化選項
    config->setMaxWorkspaceSize(1 << 30);  // 1GB
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 創建引擎
    engine_.reset(builder->buildEngineWithConfig(*network, *config));
    if (!engine_) return false;

    context_.reset(engine_->createExecutionContext());
    return context_ != nullptr;
}

void TensorRTEngine::saveEngine(const std::string& engine_path) {
    if (engine_) {
        auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
            engine_->serialize());
        std::ofstream file(engine_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(serialized->data()), 
                  serialized->size());
    }
}

bool TensorRTEngine::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;

    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return false;

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) return false;

    context_.reset(engine_->createExecutionContext());
    return context_ != nullptr;
}

cv::Mat TensorRTEngine::predict(const cv::Mat& image) {
    // 分配 GPU 記憶體
    void* gpu_input;
    void* gpu_output;
    const int input_size = INPUT_H * INPUT_W * 3 * sizeof(float);
    const int output_size = INPUT_H * INPUT_W * 6 * sizeof(float);  // 6類別
    cudaMalloc(&gpu_input, input_size);
    cudaMalloc(&gpu_output, output_size);

    // 預處理並複製到 GPU
    preprocess(image, static_cast<float*>(gpu_input));

    // 執行推理
    void* bindings[] = {gpu_input, gpu_output};
    context_->executeV2(bindings);

    // 獲取結果
    std::vector<float> output(INPUT_H * INPUT_W * 6);
    cudaMemcpy(output.data(), gpu_output, output_size, cudaMemcpyDeviceToHost);

    // 清理 GPU 記憶體
    cudaFree(gpu_input);
    cudaFree(gpu_output);

    // 後處理
    return postprocess(output.data());
}

void TensorRTEngine::preprocess(const cv::Mat& input, float* gpu_input) {
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(INPUT_W, INPUT_H));
    
    // 轉換為 float 並歸一化到 [0,1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0/255.0);
    
    // 創建 CPU 緩衝區並進行標準化
    std::vector<float> input_data(INPUT_H * INPUT_W * 3);
    float* input_ptr = input_data.data();
    
    for (int h = 0; h < INPUT_H; h++) {
        for (int w = 0; w < INPUT_W; w++) {
            for (int c = 0; c < 3; c++) {
                float pixel = float_img.at<cv::Vec3f>(h, w)[c];
                // NHWC to NCHW
                input_ptr[c * INPUT_H * INPUT_W + h * INPUT_W + w] = 
                    (pixel - mean_[c]) / std_[c];
            }
        }
    }
    
    // 複製到 GPU
    cudaMemcpy(gpu_input, input_data.data(), 
               INPUT_H * INPUT_W * 3 * sizeof(float), 
               cudaMemcpyHostToDevice);
}

cv::Mat TensorRTEngine::postprocess(float* output) {
    cv::Mat result(INPUT_H, INPUT_W, CV_8UC3);
    
    // 為每個類別創建顏色映射
    const std::vector<cv::Vec3b> colors = {
        cv::Vec3b(0, 0, 0),      // 背景
        cv::Vec3b(255, 0, 0),    // 缺陷類型 1
        cv::Vec3b(0, 255, 0),    // 缺陷類型 2
        cv::Vec3b(0, 0, 255),    // 缺陷類型 3
        cv::Vec3b(255, 255, 0),  // 缺陷類型 4
        cv::Vec3b(0, 255, 255)   // 缺陷類型 5
    };
    
    // 對每個像素找到最大概率的類別
    for (int h = 0; h < INPUT_H; h++) {
        for (int w = 0; w < INPUT_W; w++) {
            float max_prob = -1;
            int max_class = 0;
            
            // 檢查每個類別的概率
            for (int c = 0; c < 6; c++) {
                float prob = output[c * INPUT_H * INPUT_W + h * INPUT_W + w];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_class = c;
                }
            }
            
            // 設置對應的顏色
            result.at<cv::Vec3b>(h, w) = colors[max_class];
        }
    }
    
    return result;
} 