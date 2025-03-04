#pragma once

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>
#include <vector>
#include <string>

// Objective-C CoreML wrapper
@interface CoreMLObjCEngine : NSObject

@property (nonatomic, strong) MLModel *model;
@property (nonatomic, strong) NSArray<NSString *> *classNames;
@property (nonatomic, assign) NSUInteger numClasses;
@property (nonatomic, strong) MLModelDescription *modelDescription;
@property (nonatomic, strong) MLFeatureDescription *inputDescription;
@property (nonatomic, strong) MLFeatureDescription *outputDescription;

- (instancetype)initWithModelPath:(NSString *)path;
- (std::vector<std::string>)getClassNames;

@end

// MLFeatureProvider wrapper
@interface MLFeatureProviderWrapper : NSObject <MLFeatureProvider>
@property (nonatomic, strong) MLFeatureValue* imageFeature;
@property (nonatomic, strong) NSString* inputFeatureName;
@end 