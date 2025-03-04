#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreImage/CoreImage.h>
#import <Vision/Vision.h>

// 取消定義 NO 宏
#undef NO

// 然後包含其他頭文件
#include "coreml_engine.h"
#include "model_info.h"
#include <filesystem>

// 避免使用完整的 opencv.hpp
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#import "coreml_objc_engine.h"

// COMPLETELY REMOVE THE INTERFACE DEFINITION, DON'T JUST COMMENT IT OUT

static cv::Mat postprocess(MLMultiArray* multiArray);
@implementation MLFeatureProviderWrapper
- (instancetype)initWithFeatureName:(NSString *)featureName {
    self = [super init];
    if (self) {
        self.inputFeatureName = featureName;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithObject:self.inputFeatureName];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:self.inputFeatureName]) {
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

@implementation CoreMLObjCEngine

- (instancetype)initWithModelPath:(NSString *)path {
    self = [super init];
    if (self) {
        NSFileManager *fileManager = [NSFileManager defaultManager];
        
        // Check if a compiled version already exists
        NSString *compiledPath = [path stringByAppendingString:@"c"];
        BOOL isCompiledModelExists = [fileManager fileExistsAtPath:compiledPath];
        
        NSLog(@"Original model path: %@", path);
        NSLog(@"Checking for compiled model at: %@", compiledPath);
        
        NSURL *modelURL;
        if (isCompiledModelExists) {
            NSLog(@"Found existing compiled model, trying to load it first");
            modelURL = [NSURL fileURLWithPath:compiledPath];
        } else {
            NSLog(@"No compiled model found, using original path");
            modelURL = [NSURL fileURLWithPath:path];
        }
        
        // Check if file exists
        if (![fileManager fileExistsAtPath:modelURL.path]) {
            NSLog(@"ERROR: Model file does not exist at path: %@", modelURL.path);
            return nil;
        }
        
        NSError *error = nil;
        
        // Try to load the model
        NSLog(@"Attempting to load model from: %@", modelURL.path);
        _model = [MLModel modelWithContentsOfURL:modelURL error:&error];
        
        // If loading fails and we were trying the compiled version, fall back to original
        if (error && isCompiledModelExists) {
            NSLog(@"Failed to load compiled model, falling back to original");
            modelURL = [NSURL fileURLWithPath:path];
            error = nil;
            _model = [MLModel modelWithContentsOfURL:modelURL error:&error];
        }
        
        // If loading still fails, try to compile it
        if (error) {
            NSLog(@"Error loading model: %@", error.localizedDescription);
            
            NSLog(@"Attempting to compile model...");
            // Compile the model
            NSURL *compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
            if (error) {
                NSLog(@"Error compiling model: %@", error.localizedDescription);
                return nil;
            }

            // Create a permanent path for the compiled model in the same directory as the original model
            NSString *compiledModelName = [modelURL.lastPathComponent stringByAppendingString:@"c"];
            NSURL *destinationURL = [[modelURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:compiledModelName];

            // Copy the compiled model to the permanent location
            NSError *copyError = nil;

            // Remove any existing compiled model at the destination
            if ([fileManager fileExistsAtPath:destinationURL.path]) {
                NSLog(@"Removing existing compiled model at %@", destinationURL.path);
                [fileManager removeItemAtURL:destinationURL error:&copyError];
                if (copyError) {
                    NSLog(@"Warning: Failed to remove existing compiled model: %@", copyError.localizedDescription);
                    // Continue anyway
                    copyError = nil;
                }
            }

            // Copy the compiled model to the permanent location
            BOOL copySuccess = [fileManager copyItemAtURL:compiledURL toURL:destinationURL error:&copyError];
            if (copySuccess) {
                NSLog(@"Successfully copied compiled model to %@", destinationURL.path);
                
                // Load the model from the permanent location instead of the temporary one
                _model = [MLModel modelWithContentsOfURL:destinationURL error:&error];
                if (error) {
                    NSLog(@"Error loading from permanent location, falling back to temporary: %@", error.localizedDescription);
                    // If loading from the permanent location fails, fall back to the temporary one
                    _model = [MLModel modelWithContentsOfURL:compiledURL error:&error];
                    if (error) {
                        NSLog(@"Error loading compiled model: %@", error.localizedDescription);
                        return nil;
                    }
                }
            } else {
                NSLog(@"Failed to copy compiled model: %@", copyError.localizedDescription);
                // If copying fails, still try to load from the temporary location
                _model = [MLModel modelWithContentsOfURL:compiledURL error:&error];
                if (error) {
                    NSLog(@"Error loading compiled model: %@", error.localizedDescription);
                    return nil;
                }
            }
            
            NSLog(@"Successfully compiled and loaded model");
        } else {
            NSLog(@"Model loaded successfully%@", isCompiledModelExists ? @" from compiled version" : @" from original version");
        }
        
        // Extract class names from model metadata if available
        NSDictionary *metadata = nil;
        @try {
            // First try the modern API
            if ([_model.modelDescription.metadata respondsToSelector:@selector(userDefined)]) {
                metadata = [_model.modelDescription.metadata valueForKey:@"userDefined"];
            } 
            // Then try the older API
            else if ([_model.modelDescription.metadata respondsToSelector:@selector(userDefinedMetadata)]) {
                metadata = [_model.modelDescription.metadata valueForKey:@"userDefinedMetadata"];
            }
            // If neither works, try direct dictionary access
            else if ([_model.modelDescription.metadata isKindOfClass:[NSDictionary class]]) {
                metadata = (NSDictionary *)_model.modelDescription.metadata;
            }
        } @catch (NSException *exception) {
            NSLog(@"Error accessing metadata: %@", exception);
            metadata = @{};
        }
        
        if (metadata && metadata[@"class_names"]) {
            NSString *classNamesStr = metadata[@"class_names"];
            _classNames = [classNamesStr componentsSeparatedByString:@","];
            _numClasses = _classNames.count;
            NSLog(@"Loaded model with %lu classes from metadata", (unsigned long)_numClasses);
        } else {
            // Default class names if not in metadata
            _classNames = @[@"OK", @"pitted_surface", @"inclusion", @"patches", 
                           @"rolled-in_scale", @"scratches", @"crazing"];
            _numClasses = _classNames.count;
            NSLog(@"Using default class names, %lu classes", (unsigned long)_numClasses);
        }
        
        // Create the model input and output descriptions
        _inputDescription = _model.modelDescription.inputDescriptionsByName[@"input_image"];
        _outputDescription = _model.modelDescription.outputDescriptionsByName[@"output"];
    }
    return self;
}

- (std::vector<std::string>)getClassNames {
    // Convert NSArray of class names to std::vector<std::string>
    std::vector<std::string> result;
    for (NSString *name in _classNames) {
        result.push_back(std::string([name UTF8String]));
    }
    return result;
}

- (cv::Mat)predict:(const cv::Mat&)image {
    // Convert OpenCV image to CoreML compatible format
    NSLog(@"Model input features: %@", _model.modelDescription.inputDescriptionsByName.allKeys);
    
    // Find the first image input feature
    NSString *inputFeatureName = nil;
    for (NSString *key in _model.modelDescription.inputDescriptionsByName.allKeys) {
        MLFeatureDescription *desc = _model.modelDescription.inputDescriptionsByName[key];
        if (desc.type == MLFeatureTypeImage) {
            inputFeatureName = key;
            break;
        }
    }
    
    if (!inputFeatureName) {
        NSLog(@"No image input feature found");
        return cv::Mat();
    }
    
    NSLog(@"Using input feature: %@", inputFeatureName);
    
    // Convert OpenCV Mat to CGImage
    CGImageRef cgImage = [self CGImageFromCVMat:image];
    if (!cgImage) {
        NSLog(@"Failed to convert OpenCV image to CGImage");
        return cv::Mat();
    }
    
    // Create an MLFeatureValue from the CGImage
    NSError *error = nil;
    MLFeatureValue *imageFeature = [MLFeatureValue featureValueWithCGImage:cgImage 
                                                               pixelsWide:image.cols 
                                                               pixelsHigh:image.rows 
                                                                pixelFormatType:kCVPixelFormatType_32BGRA 
                                                                options:nil 
                                                                error:&error];
    if (error) {
        NSLog(@"Error creating feature value: %@", error);
        CGImageRelease(cgImage);
        return cv::Mat();
    }
    
    // Create a dictionary of input features
    NSDictionary<NSString *, MLFeatureValue *> *inputFeatures = @{
        inputFeatureName: imageFeature
    };
    
    // Create an MLDictionaryFeatureProvider
    MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputFeatures error:&error];
    if (error) {
        NSLog(@"Error creating feature provider: %@", error);
        CGImageRelease(cgImage);
        return cv::Mat();
    }
    
    // Make prediction
    id<MLFeatureProvider> outputFeatures = [_model predictionFromFeatures:provider error:&error];
    if (error) {
        NSLog(@"Prediction error: %@", error);
        CGImageRelease(cgImage);
        return cv::Mat();
    }
    
    // Release the CGImage
    CGImageRelease(cgImage);
    
    // Log available output features
    NSLog(@"Model output features: %@", _model.modelDescription.outputDescriptionsByName.allKeys);
    
    // Try to get the output feature - first try "output", then try the first available output
    MLFeatureValue *outputFeature = [outputFeatures featureValueForName:@"output"];
    if (!outputFeature) {
        // If "output" is not found, try the first output feature
        NSString *firstOutputName = _model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
        if (firstOutputName) {
            NSLog(@"Output feature %@: type=%d", firstOutputName, (int)[outputFeatures featureValueForName:firstOutputName].type);
            outputFeature = [outputFeatures featureValueForName:firstOutputName];
        }
    }
    
    if (!outputFeature) {
        NSLog(@"Output feature not found");
        return cv::Mat();
    }
    
    // Check if the output feature is a multi-array
    if (outputFeature.type != MLFeatureTypeMultiArray) {
        NSLog(@"Output feature is not a multi-array, type=%d", (int)outputFeature.type);
        return cv::Mat();
    }
    
    // Check if the multi-array is valid
    if (!outputFeature.multiArrayValue) {
        NSLog(@"Output array is null");
        return cv::Mat();
    }
    
    // Convert MLMultiArray to cv::Mat
    return [self cvMatFromMultiArray:outputFeature.multiArrayValue];
}

- (CGImageRef)CGImageFromCVMat:(const cv::Mat&)cvMat {
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize() * cvMat.total()];
    
    CGColorSpaceRef colorSpace;
    if (cvMat.channels() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    CGBitmapInfo bitmapInfo;
    if (cvMat.channels() == 1) {
        bitmapInfo = kCGImageAlphaNone;
    } else if (cvMat.channels() == 3) {
        bitmapInfo = kCGImageAlphaNone | kCGBitmapByteOrderDefault;
    } else {
        bitmapInfo = kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault;
    }
    
    CGImageRef imageRef = CGImageCreate(
        cvMat.cols,                 // width
        cvMat.rows,                 // height
        8,                          // bits per component
        8 * cvMat.channels(),       // bits per pixel
        cvMat.step[0],              // bytes per row
        colorSpace,                 // color space
        bitmapInfo,                 // bitmap info
        provider,                   // data provider
        NULL,                       // decode
        false,                      // should interpolate
        kCGRenderingIntentDefault   // intent
    );
    
    CGColorSpaceRelease(colorSpace);
    CGDataProviderRelease(provider);
    
    return imageRef;
}

- (cv::Mat)cvMatFromMultiArray:(MLMultiArray *)multiArray {
    // Add more debugging
    NSLog(@"MultiArray shape: %@", multiArray.shape);
    NSLog(@"MultiArray dataType: %ld", (long)multiArray.dataType);
    NSLog(@"MultiArray count: %ld", (long)multiArray.count);
    
    // Get dimensions
    NSInteger count = multiArray.count;
    NSInteger channels = 1;  // Default for segmentation masks
    NSInteger height = 1;
    NSInteger width = 1;
    
    // Determine dimensions based on shape
    if (multiArray.shape.count == 4) {
        // Format is likely [batch, channels, height, width]
        channels = [multiArray.shape[1] integerValue];
        height = [multiArray.shape[2] integerValue];
        width = [multiArray.shape[3] integerValue];
    } else if (multiArray.shape.count == 3) {
        // Format is likely [channels, height, width]
        channels = [multiArray.shape[0] integerValue];
        height = [multiArray.shape[1] integerValue];
        width = [multiArray.shape[2] integerValue];
    } else if (multiArray.shape.count == 2) {
        // Format is likely [height, width]
        height = [multiArray.shape[0] integerValue];
        width = [multiArray.shape[1] integerValue];
    }
    
    // Create OpenCV Mat
    cv::Mat result;
    
    // For segmentation, we typically want the argmax across channels
    if (channels > 1) {
        // Create a matrix to hold the class predictions
        result = cv::Mat::zeros(height, width, CV_8UC1);
        
        // For each pixel, find the channel with the highest value (argmax)
        for (NSInteger y = 0; y < height; y++) {
            for (NSInteger x = 0; x < width; x++) {
                float maxVal = -INFINITY;
                int maxIdx = 0;
                
                // Find the channel with the highest value
                for (NSInteger c = 0; c < channels; c++) {
                    NSArray<NSNumber *> *indices;
                    if (multiArray.shape.count == 4) {
                        indices = @[@0, @(c), @(y), @(x)];
                    } else {
                        indices = @[@(c), @(y), @(x)];
                    }
                    
                    float val = [multiArray[indices] floatValue];
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = (int)c;
                    }
                }
                
                // Set the pixel value to the class index
                result.at<uchar>(y, x) = (uchar)maxIdx;
            }
        }
    } else {
        // If there's only one channel, just copy the data
        result = cv::Mat(height, width, CV_32FC1);
        
        for (NSInteger y = 0; y < height; y++) {
            for (NSInteger x = 0; x < width; x++) {
                NSArray<NSNumber *> *indices;
                if (multiArray.shape.count == 4) {
                    indices = @[@0, @0, @(y), @(x)];
                } else if (multiArray.shape.count == 3) {
                    indices = @[@0, @(y), @(x)];
                } else {
                    indices = @[@(y), @(x)];
                }
                
                result.at<float>(y, x) = [multiArray[indices] floatValue];
            }
        }
    }
    
    return result;
}

@end

CoreMLEngine::CoreMLEngine(const std::string& model_path)
    : objc_engine_(nullptr),
      model_info_(findModelInfoPath(model_path)) {
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:model_path.c_str()];
        CoreMLObjCEngine* engine = [[CoreMLObjCEngine alloc] initWithModelPath:path];
        if (!engine) {
            throw std::runtime_error("Failed to initialize CoreML engine");
        }
        objc_engine_ = (void*)CFBridgingRetain(engine);
    }
}

CoreMLEngine::~CoreMLEngine() {
    if (objc_engine_) {
        CFBridgingRelease(objc_engine_);
        objc_engine_ = nullptr;
    }
}

const ModelInfo& CoreMLEngine::getModelInfo() const {
    return model_info_;
}

std::string CoreMLEngine::findModelInfoPath(const std::string& model_path) {
    // Look for modelInfo.inf in the same directory as the model
    std::filesystem::path model_dir = std::filesystem::path(model_path).parent_path();
    std::string info_path = (model_dir / "modelInfo.inf").string();
        
    if (std::filesystem::exists(info_path)) {
        return info_path;
    }
    
    // Try current directory
    info_path = "../modelInfo.inf";
    if (std::filesystem::exists(info_path)) {
        return info_path;
    }
    
    throw std::runtime_error("Could not find modelInfo.inf file");
}

cv::Mat CoreMLEngine::predict(const cv::Mat& image) {
    @autoreleasepool {
        MLModelWrapper* wrapper = (__bridge MLModelWrapper*)objc_engine_;
        NSError* error = nil;
        
        // 打印輸入圖像尺寸和通道數
        NSLog(@"Input image shape: %dx%d, channels: %d", image.rows, image.cols, image.channels());
        
        // 預處理
        cv::Mat preprocessed = preprocess(image);
        
        // Instead of converting to BGRA, ensure we're using BGR format
        cv::Mat bgr;
        if (preprocessed.channels() == 1) {
            cv::cvtColor(preprocessed, bgr, cv::COLOR_GRAY2BGR);
        } else if (preprocessed.channels() == 4) {
            cv::cvtColor(preprocessed, bgr, cv::COLOR_BGRA2BGR);
        } else {
            bgr = preprocessed;
        }
        
        NSLog(@"Input image shape: %dx%d, channels: %d", bgr.rows, bgr.cols, bgr.channels());
        
        // 將 OpenCV Mat 轉換為 CGImage
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        NSData* data = [NSData dataWithBytes:bgr.data 
                                    length:bgr.total() * bgr.elemSize()];
        CGDataProviderRef dataProvider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
            
        CGImageRef imageRef = CGImageCreate(
            bgr.cols,
            bgr.rows,
            8,
            8 * bgr.channels(),  // Number of bits depends on channels
            bgr.step,
            colorSpace,
            kCGBitmapByteOrderDefault | kCGImageAlphaNone,  // No alpha channel for BGR
            dataProvider,
            NULL,
            false,
            kCGRenderingIntentDefault);
            
        // 創建 CIImage
        CIImage* ciImage = [CIImage imageWithCGImage:imageRef];
        
        // 使用 CIImage 創建 CGImageRef 作為 MLFeatureValue 輸入
        MLFeatureValue* imageFeatureValue = [MLFeatureValue featureValueWithCGImage:imageRef 
                                                                    pixelsWide:bgr.cols 
                                                                    pixelsHigh:bgr.rows 
                                                                    pixelFormatType:kCVPixelFormatType_24BGR 
                                                                    options:nil 
                                                                    error:&error];
        
        if (error) {
            NSLog(@"Error creating feature value: %@", error);
            CGImageRelease(imageRef);
            CGDataProviderRelease(dataProvider);
            CGColorSpaceRelease(colorSpace);
            throw std::runtime_error("Failed to create feature value");
        }
        
        // 打印模型輸入信息
        NSDictionary* inputDesc = wrapper.model.modelDescription.inputDescriptionsByName;
        NSLog(@"Model input features: %@", inputDesc.allKeys);
        
        // 找到第一個圖像類型的輸入特徵
        NSString* inputFeatureName = nil;
        for (NSString* key in inputDesc.allKeys) {
            MLFeatureDescription* desc = inputDesc[key];
            if (desc.type == MLFeatureTypeImage) {
                inputFeatureName = key;
                break;
            }
        }

        if (!inputFeatureName) {
            NSLog(@"No image input feature found");
            throw std::runtime_error("No suitable input feature found");
        }

        NSLog(@"Using input feature: %@", inputFeatureName);

        // 創建輸入特徵提供者
        MLFeatureProviderWrapper* featureProvider = [[MLFeatureProviderWrapper alloc] initWithFeatureName:inputFeatureName];
        featureProvider.imageFeature = imageFeatureValue;

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
        
        // 在獲取輸出之前，打印所有可用的輸出特徵
        for (NSString* key in outputFeatures.featureNames) {
            MLFeatureValue* value = [outputFeatures featureValueForName:key];
            NSLog(@"Output feature %@: type=%ld", key, (long)value.type);
        }
        
        // 獲取輸出
        NSString* outputFeatureName = nil;
        for (NSString* key in outputFeatures.featureNames) {
            MLFeatureValue* value = [outputFeatures featureValueForName:key];
            if (value.type == MLFeatureTypeMultiArray) {
                outputFeatureName = key;
                break;
            }
        }

        if (!outputFeatureName) {
            NSLog(@"No multiarray output feature found");
            throw std::runtime_error("No suitable output feature found");
        }

        NSLog(@"Using output feature: %@", outputFeatureName);
        MLMultiArray* outputArray = (MLMultiArray*)[outputFeatures 
            featureValueForName:outputFeatureName].multiArrayValue;

        if (!outputArray) {
            NSLog(@"Output array is null");
            throw std::runtime_error("Failed to get output array");
        }
        
        // 打印輸出數組的詳細信息
        NSLog(@"Output array shape: %@", outputArray.shape);
        NSLog(@"Output array dataType: %ld", (long)outputArray.dataType);
        NSLog(@"Output array count: %ld", (long)outputArray.count);
        
        // 後處理 - directly pass the MLMultiArray
        cv::Mat result = postprocess(outputArray);
        
        // 清理資源
        CGImageRelease(imageRef);
        CGDataProviderRelease(dataProvider);
        CGColorSpaceRelease(colorSpace);
        
        return result;
    }
}

cv::Mat CoreMLEngine::preprocess(const cv::Mat& input) {
    // 確保輸入是 RGB 圖像
    cv::Mat rgb;
    if (input.channels() == 1) {
        cv::cvtColor(input, rgb, cv::COLOR_GRAY2BGR);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, rgb, cv::COLOR_BGRA2BGR);
    } else {
        rgb = input;
    }
    
    // 調整大小
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
    
    // 轉換為浮點數並歸一化
    return resized;
}

// Add these helper functions for Float16 conversion at the top of the file
// IEEE-754 half-precision floating point conversion
static inline float Float16ToFloat32(uint16_t h) {
    // Extract components
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    
    // Handle special cases
    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;  // +/- zero
        // Denormalized number
        float result = sign ? -1.0f : 1.0f;
        result *= (float)frac / 1024.0f;
        result *= 1.0f / 16384.0f;
        return result;
    } else if (exp == 31) {
        if (frac == 0) return sign ? -INFINITY : INFINITY;  // +/- infinity
        return NAN;  // NaN
    }
    
    // Normalized number
    uint32_t f = 0;
    f |= sign << 31;                       // Sign bit
    f |= (exp - 15 + 127) << 23;           // Exponent, adjusted for bias
    f |= frac << 13;                       // Fraction, shifted left
    
    return *((float*)&f);  // Bit-cast to float
}

static inline uint16_t Float16_NInf()
{
    return 0xFC00;  // Correct representation for -∞ in Float16
}
static inline uint16_t Float16_Inf()
{
    return 0x7C00;
}

static inline bool Float16_Bigger(uint16_t f1, uint16_t f2) {
    // Extract components
    uint32_t sign1 = (f1 >> 15) & 0x1;
    
    uint32_t sign2 = (f2 >> 15) & 0x1;

    // Handle sign cases
    if (sign1 != sign2) {
        return sign1 < sign2;  // Positive is always greater than negative
    }

    uint32_t exp1 = (f1 >> 10) & 0x1F;
    uint32_t exp2 = (f2 >> 10) & 0x1F;
    // If both are positive, compare normally
    // If both are negative, the comparison is reversed
    if (exp1 != exp2) {
        return (sign1 == 1) ^(exp1 > exp2);
    }

    uint32_t frac1 = f1 & 0x3FF;
    uint32_t frac2 = f2 & 0x3FF;
    if (frac1 != frac2) {
        return (sign1 == 1)^(frac1 > frac2);
    }

    return false;  // Both values are equal
}


// Modify the postprocess function to use direct buffer access for Float16 data
cv::Mat postprocess(MLMultiArray* multiArray) {
    if (!multiArray) {
        printf("Error: MLMultiArray is null\n");
        return cv::Mat();
    }
    
    // NSLog(@"MultiArray Data Type: %ld (0x%lX)", (long)multiArray.dataType, (long)multiArray.dataType);
    
    // Get dimensions from multiArray shape
    NSUInteger batchSize = [multiArray.shape[0] unsignedIntegerValue];
    NSUInteger channels = [multiArray.shape[1] unsignedIntegerValue];
    NSUInteger height = [multiArray.shape[2] unsignedIntegerValue];
    NSUInteger width = [multiArray.shape[3] unsignedIntegerValue];
    
    // NSLog(@"Processing shape: [%lu,%lu,%lu,%lu]", batchSize, channels, height, width);
    
    // Create result matrix
    cv::Mat result(height, width, CV_8UC3);


    
    // Define colors for each class
    const cv::Vec3b colors[] = {
        cv::Vec3b(0, 0, 0),      // Background/OK 
        cv::Vec3b(255, 0, 0),    // pitted_surface
        cv::Vec3b(0, 255, 0),    // inclusion
        cv::Vec3b(0, 0, 255),    // patches
        cv::Vec3b(255, 255, 0),  // rolled-in_scale
        cv::Vec3b(0, 255, 255),  // scratches
        cv::Vec3b(255, 0, 255)   // crazing
    };
    
    int MAX_COLORS = sizeof(colors) / sizeof(colors[0]);
    
    if (multiArray.dataType == MLMultiArrayDataTypeFloat16) {
        // NSLog(@"Using direct Float16 buffer access");
        
        // Get raw data pointer
        void* dataPtr = multiArray.dataPointer;
        uint16_t* float16Ptr = (uint16_t*)dataPtr;
        
        // Get strides
        NSArray<NSNumber *>* strides = multiArray.strides;
        NSUInteger batchStride = [strides[0] unsignedIntegerValue];
        NSUInteger channelStride = [strides[1] unsignedIntegerValue];
        NSUInteger rowStride = [strides[2] unsignedIntegerValue];
        NSUInteger colStride = [strides[3] unsignedIntegerValue];
        
        // NSLog(@"Strides: [%lu,%lu,%lu,%lu]", batchStride, channelStride, rowStride, colStride);
        
        
        if(0)
        {//test method
            cv::Mat class_buf(height, width, CV_32SC1);
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    NSUInteger baseIdx = h * rowStride + w * colStride;
                    
                    class_buf.at<uint32_t>(h, w) =float16Ptr[baseIdx];

                }
            }
            
            for (int c = 1; c < channels; c++) {
                int chAdd=c*channelStride;
                for (int h = 0; h < height; h++) {
                    int hAdd=h*rowStride;
                    for (int w = 0; w < width; w++) {
                        uint32_t max_class_prob=class_buf.at<uint32_t>(h,w);
                        uint16_t curMaxProb=max_class_prob&0xFFFF;
                        int maxClass=max_class_prob>>16;
                        NSUInteger idx = hAdd + w * colStride + chAdd;
                        uint16_t prob=float16Ptr[idx];
                        if (Float16_Bigger(prob, max_class_prob)) {
                            class_buf.at<uint32_t>(h,w)=(c<<16)+prob;
                            maxClass=c;
                        }

                        if(c==channels-1){
                            result.at<cv::Vec3b>(h,w)=(maxClass>=0&&maxClass<MAX_COLORS)?colors[maxClass]:cv::Vec3b(128,128,128);
                        }
                    }
                }
            }
        }



        // Process row-by-row to minimize cache misses
        if(1)for (int h = 0; h < height; h++) {
            // if (h % 25 == 0) {
            //     printf("Processing row %d/%lu\n", h, height);
            // }
            
            for (int w = 0; w < width; w++) {
                // float maxProb = -INFINITY;
                uint16_t maxProb=Float16_NInf();
                int maxClass = 0;
                
                // Base index for this pixel (batch index is always 0)
                NSUInteger baseIdx = h * rowStride + w * colStride;
                
                // Find max class across all channels
                for (int c = 0; c < channels; c++) {
                    NSUInteger idx = baseIdx + c * channelStride;
                    
                    // Convert Float16 to Float32
                    uint16_t prob = float16Ptr[idx];
                    
                    if (Float16_Bigger(prob, maxProb)) {
                        maxProb = prob;
                        maxClass = c;
                    }
                }
                
                // Set color based on max class
                result.at<cv::Vec3b>(h, w) = (maxClass >= 0 && maxClass < MAX_COLORS) ? 
                                            colors[maxClass] : cv::Vec3b(128, 128, 128);
            }
        }
    } else {
        NSLog(@"Using standard access method (non-Float16)");
        
        // Fall back to original implementation for non-Float16 data
        for (int h = 0; h < height; h++) {
            if (h % 25 == 0) {
                printf("Processing row %d/%lu\n", h, height);
            }
            
            for (int w = 0; w < width; w++) {
                float maxProb = -INFINITY;
                int maxClass = 0;
                
                for (int c = 0; c < channels; c++) {
                    NSArray<NSNumber *> *indices = @[@0, @(c), @(h), @(w)];
                    float prob = [multiArray[indices] floatValue];
                    
                    if (prob > maxProb) {
                        maxProb = prob;
                        maxClass = c;
                    }
                }
                
                result.at<cv::Vec3b>(h, w) = (maxClass >= 0 && maxClass < MAX_COLORS) ? 
                                           colors[maxClass] : cv::Vec3b(128, 128, 128);
            }
        }
    }
    
    return result;
}

// cv::Mat CoreMLEngine::predict_batch(const std::vector<cv::Mat>& images) {
//     if (images.empty()) {
//         return cv::Mat();
//     }
//     
//     // Process images with separate autoreleasepool for each image
//     cv::Mat result;
//     MLModelWrapper* wrapper = (__bridge MLModelWrapper*)objc_engine_;
//     
//     // Create options once
//     MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
//     
//     // Track memory usage
//     static int image_counter = 0;
//     
//     for (size_t i = 0; i < images.size(); i++) {
//         @autoreleasepool {
//             try {
//                 // Count total images processed
//                 image_counter++;
//                 
//                 // Every 20 images, create a fresh options object
//                 if (image_counter % 20 == 0) {
//                     options = [[MLPredictionOptions alloc] init];
//                 }
//                 
//                 // ... rest of the method ...
//             }
//         }
//     }
//     
//     return result;
// } 