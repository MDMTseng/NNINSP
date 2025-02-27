#include "inference_engine.h"
#ifdef __APPLE__
#include "coreml_engine.h"
#else
#include "tensorrt_engine.h"
#endif

std::unique_ptr<InferenceEngine> InferenceEngine::create(const std::string& model_path) {
#ifdef __APPLE__
    return std::make_unique<CoreMLEngine>(model_path);
#else
    return std::make_unique<TensorRTEngine>(model_path);
#endif
} 