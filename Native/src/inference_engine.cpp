#include "inference_engine.h"

#ifdef __APPLE__
// Forward declare the CoreMLEngine creation function
std::unique_ptr<InferenceEngine> createCoreMLEngine(const std::string& model_path);
#else
#include "tensorrt_engine.h"
#endif

std::unique_ptr<InferenceEngine> InferenceEngine::create(const std::string& model_path) {
#ifdef __APPLE__
    return createCoreMLEngine(model_path);
#else
    return std::make_unique<TensorRTEngine>(model_path);
#endif
} 