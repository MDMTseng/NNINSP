#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreImage/CoreImage.h>
#import <Vision/Vision.h>

// 取消定義 NO 宏
#undef NO

// 然後包含其他頭文件
#include "coreml_engine.h"

// 避免使用完整的 opencv.hpp
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// 使用 MLFeatureProvider 包裝輸入數據
@interface MLFeatureProviderWrapper : NSObject <MLFeatureProvider>
@property (nonatomic, strong) MLFeatureValue* imageFeature;
@end

@implementation MLFeatureProviderWrapper
- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithObject:@"input"];
}
- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"input"]) {
        return self.imageFeature;
    }
    return nil;
}
@end

@interface MLModelWrapper : NSObject
@property (nonatomic, strong) MLModel* model;
@end

@implementation MLModelWrapper
@end

CoreMLEngine::CoreMLEngine(const std::string& model_path) {
    @autoreleasepool {
        NSError* error = nil;
        
        // 將路徑轉換為 NSURL
        NSString* path = [NSString stringWithUTF8String:model_path.c_str()];
        NSURL* modelURL = [NSURL fileURLWithPath:path];
        
        // 編譯模型（如果需要）
        NSURL* compiledModelURL = [MLModel compileModelAtURL:modelURL error:&error];
        if (error || !compiledModelURL) {
            NSLog(@"Error compiling model: %@", error);
            throw std::runtime_error("Failed to compile CoreML model");
        }
        
        // 載入編譯後的模型
        MLModel* model = [MLModel modelWithContentsOfURL:compiledModelURL error:&error];
        if (error || !model) {
            NSLog(@"Error loading model: %@", error);
            throw std::runtime_error("Failed to load CoreML model");
        }
        
        // 打印輸入形狀信息
        MLFeatureDescription* inputDesc = model.modelDescription.inputDescriptionsByName[@"input"];
        MLImageConstraint* imageConstraint = inputDesc.imageConstraint;
        NSLog(@"Expected input shape: %dx%d", (int)imageConstraint.pixelsHigh, (int)imageConstraint.pixelsWide);
        
        // 創建包裝器
        MLModelWrapper* wrapper = [[MLModelWrapper alloc] init];
        wrapper.model = model;
        model_ = (void*)CFBridgingRetain(wrapper);
    }
}

CoreMLEngine::~CoreMLEngine() {
    if (model_) {
        CFBridgingRelease(model_);
        model_ = nullptr;
    }
}

cv::Mat CoreMLEngine::predict(const cv::Mat& image) {
    @autoreleasepool {
        MLModelWrapper* wrapper = (__bridge MLModelWrapper*)model_;
        NSError* error = nil;
        
        // 打印輸入圖像尺寸和通道數
        // NSLog(@"Input image shape: %dx%d, channels: %d", image.rows, image.cols, image.channels());
        
        // 預處理
        cv::Mat preprocessed = preprocess(image);
        // NSLog(@"Preprocessed image shape: %dx%d, channels: %d", 
        //       preprocessed.rows, preprocessed.cols, preprocessed.channels());
        
        // 將單通道圖像轉換為 BGRA 格式
        cv::Mat bgra;
        cv::cvtColor(preprocessed, bgra, cv::COLOR_GRAY2BGRA);
        
        // 將 OpenCV Mat 轉換為 CGImage
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        NSData* data = [NSData dataWithBytes:bgra.data 
                                    length:bgra.total() * bgra.elemSize()];
        CGDataProviderRef dataProvider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
            
        CGImageRef imageRef = CGImageCreate(
            bgra.cols,
            bgra.rows,
            8,
            32,  // 4 channels * 8 bits
            bgra.step,
            colorSpace,
            kCGBitmapByteOrderDefault | kCGImageAlphaNoneSkipLast,
            dataProvider,
            NULL,
            false,
            kCGRenderingIntentDefault);
            
        // 創建 CIImage 和 CVPixelBuffer
        CIImage* ciImage = [CIImage imageWithCGImage:imageRef];
        CVPixelBufferRef pixelBuffer = NULL;
        CVPixelBufferCreate(kCFAllocatorDefault,
                          bgra.cols,
                          bgra.rows,
                          kCVPixelFormatType_32BGRA,  // 使用 BGRA 格式
                          NULL,
                          &pixelBuffer);
        
        // 將 CIImage 渲染到 CVPixelBuffer
        CIContext* context = [CIContext contextWithOptions:nil];
        [context render:ciImage toCVPixelBuffer:pixelBuffer];
        
        // 創建 MLFeatureValue
        MLFeatureValue* imageFeatureValue = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
        
        // 創建輸入特徵提供者
        MLFeatureProviderWrapper* featureProvider = [[MLFeatureProviderWrapper alloc] init];
        featureProvider.imageFeature = imageFeatureValue;
        
        // 打印模型輸入信息
        NSDictionary* inputDesc = wrapper.model.modelDescription.inputDescriptionsByName;
        NSLog(@"Model input features: %@", inputDesc.allKeys);
        
        // 運行推理
        MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
        id<MLFeatureProvider> outputFeatures = [wrapper.model 
            predictionFromFeatures:featureProvider
            options:options
            error:&error];
        
        if (error) {
            NSLog(@"Prediction error: %@", error);
            throw std::runtime_error("Prediction failed");
        }
        
        // 打印模型輸出信息
        NSDictionary* outputDesc = wrapper.model.modelDescription.outputDescriptionsByName;
        NSLog(@"Model output features: %@", outputDesc.allKeys);
        
        // 獲取輸出 - 使用正確的特徵名稱
        MLMultiArray* outputArray = (MLMultiArray*)[outputFeatures 
            featureValueForName:@"var_1573"].multiArrayValue;
        
        if (!outputArray) {
            NSLog(@"Output array is null");
            throw std::runtime_error("Failed to get output array");
        }
        
        // 打印輸出形狀
        NSLog(@"Output shape: %@", outputArray.shape);
        
        // 後處理
        cv::Mat result = postprocess((float*)outputArray.dataPointer, 
                         [outputArray.shape[1] integerValue] * 
                         [outputArray.shape[2] integerValue] * 
                         [outputArray.shape[3] integerValue]);
        
        // 清理資源
        CGImageRelease(imageRef);
        CGDataProviderRelease(dataProvider);
        CGColorSpaceRelease(colorSpace);
        CVPixelBufferRelease(pixelBuffer);
        
        return result;
    }
}

cv::Mat CoreMLEngine::preprocess(const cv::Mat& input) {
    // 確保輸入是單通道圖像
    cv::Mat gray;
    if (input.channels() > 1) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input;
    }
    
    // 調整大小
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
    
    // 轉換為浮點數並歸一化
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0/255.0);
    
    return float_img;
}

cv::Mat CoreMLEngine::postprocess(const float* output, size_t size) {
    const int OUTPUT_H = 256;  // 修改輸出尺寸
    const int OUTPUT_W = 256;
    const int OUTPUT_C = 6;
    printf("OUTPUT_H: %d, OUTPUT_W: %d  size: %d\n", OUTPUT_H, OUTPUT_W, size);
    cv::Mat result(OUTPUT_H, OUTPUT_W, CV_8UC3);
    
    // 為每個類別創建顏色映射
    const cv::Vec3b colors[] = {
        cv::Vec3b(0, 0, 0),      // 背景
        cv::Vec3b(255, 0, 0),    // 缺陷類型 1
        cv::Vec3b(0, 255, 0),    // 缺陷類型 2
        cv::Vec3b(0, 0, 255),    // 缺陷類型 3
        cv::Vec3b(255, 255, 0),  // 缺陷類型 4
        cv::Vec3b(0, 255, 255)   // 缺陷類型 5
    };
    
    // 對每個像素找到最大概率的類別
    for (int h = 0; h < OUTPUT_H; h++) {
        for (int w = 0; w < OUTPUT_W; w++) {
            float max_prob = -1;
            int max_class = 0;
            
            for (int c = 0; c < OUTPUT_C; c++) {
                float prob = output[c * OUTPUT_H * OUTPUT_W + h * OUTPUT_W + w];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_class = c;
                }
            }
            
            result.at<cv::Vec3b>(h, w) = colors[max_class];

            result.at<cv::Vec3b>(h, w) = cv::Vec3b(200*output[0 * OUTPUT_H * OUTPUT_W + h * OUTPUT_W + w],
                                                   200*output[1 * OUTPUT_H * OUTPUT_W + h * OUTPUT_W + w],
                                                   200*output[2 * OUTPUT_H * OUTPUT_W + h * OUTPUT_W + w]);
        }
    }
    
    // 將結果調整到原始大小
    cv::Mat resized;
    cv::resize(result, resized, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
    return resized;
} 