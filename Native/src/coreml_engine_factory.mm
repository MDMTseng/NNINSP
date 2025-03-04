#include "inference_engine.h"
#include "coreml_engine.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

// Add this function to get absolute path
NSString* getAbsolutePath(const std::string& relativePath) {
    NSString* relPath = [NSString stringWithUTF8String:relativePath.c_str()];
    if ([relPath hasPrefix:@"/"]) {
        return relPath; // Already absolute
    }
    
    NSString* currentDir = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString* absPath = [currentDir stringByAppendingPathComponent:relPath];
    return [absPath stringByStandardizingPath];
}

// This function is called from inference_engine.cpp
std::unique_ptr<InferenceEngine> createCoreMLEngine(const std::string& model_path) {
    // Get absolute path
    NSString *nsModelPath = getAbsolutePath(model_path);
    
    NSLog(@"AA");
    // Check if we have a pre-compiled model
    NSString *compiledPath = [nsModelPath stringByAppendingString:@"c"];  // .mlpackage -> .mlpackagec
    if ([[NSFileManager defaultManager] fileExistsAtPath:compiledPath]) {
        NSLog(@"Using pre-compiled model at: %@", compiledPath);
        return std::make_unique<CoreMLEngine>([compiledPath UTF8String]);
    }
    
    NSURL *modelURL = [NSURL fileURLWithPath:nsModelPath];
    
    NSLog(@"[DEBUG] Attempting to load CoreML model from path: %@", nsModelPath);
    
    // Check if the model is a directory
    BOOL isDirectory;
    if ([[NSFileManager defaultManager] fileExistsAtPath:nsModelPath isDirectory:&isDirectory] && isDirectory) {
        NSLog(@"[DEBUG] Model path is a directory, attempting to load directly without compilation");
        
        // List directory contents to verify it's a valid .mlpackage
        NSError *listError = nil;
        NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:nsModelPath error:&listError];
        if (listError) {
            NSLog(@"Error listing directory contents: %@", listError);
        } else {
            NSLog(@"Directory contents: %@", contents);
        }
        
        // Try to load the model directly without compilation
        return std::make_unique<CoreMLEngine>([nsModelPath UTF8String]);
    } else {
        NSLog(@"[DEBUG] Model path is not a directory or doesn't exist, trying direct load");
    }
    
    // If it's not a directory or doesn't exist, try to load it directly
    return std::make_unique<CoreMLEngine>(model_path);
} 