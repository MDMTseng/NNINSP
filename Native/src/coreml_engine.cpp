// cv::Mat CoreMLEngine::predict(const cv::Mat& image) {
//     cv::Mat result;
//     @autoreleasepool {
//         CoreMLObjCEngine* engine = (__bridge CoreMLObjCEngine*)objc_engine_;
//         result = [engine predict:image];
        
//         // Check if the result is empty
//         if (result.empty()) {
//             throw std::runtime_error("Prediction failed to produce valid output");
//         }
//     }
//     return result;
// }

// cv::Mat CoreMLEngine::postprocess(const float* output, size_t size) {
//     // Get dimensions from the size
//     const int OUTPUT_H = 256;
//     const int OUTPUT_W = 256;
//     const int OUTPUT_C = 7;  // Changed from 6 to 7 to match model output
    
//     printf("OUTPUT_H: %d, OUTPUT_W: %d, OUTPUT_C: %d, size: %zu\n", OUTPUT_H, OUTPUT_W, OUTPUT_C, size);
    
//     // Verify that the size matches our expectations
//     if (size != OUTPUT_H * OUTPUT_W * OUTPUT_C) {
//         printf("Warning: Output size mismatch. Expected %d, got %zu\n", 
//                OUTPUT_H * OUTPUT_W * OUTPUT_C, size);
//     }
    
//     cv::Mat result(OUTPUT_H, OUTPUT_W, CV_8UC3);
    
//     // Define colors for each class
//     const cv::Vec3b colors[] = {
//         cv::Vec3b(0, 0, 0),      // Background/OK
//         cv::Vec3b(255, 0, 0),    // pitted_surface
//         cv::Vec3b(0, 255, 0),    // inclusion
//         cv::Vec3b(0, 0, 255),    // patches
//         cv::Vec3b(255, 255, 0),  // rolled-in_scale
//         cv::Vec3b(0, 255, 255),  // scratches
//         cv::Vec3b(255, 0, 255)   // crazing
//     };
    
//     // For each pixel, find the class with highest probability
//     for (int h = 0; h < OUTPUT_H; h++) {
//         for (int w = 0; w < OUTPUT_W; w++) {
//             float max_prob = -1;
//             int max_class = 0;
            
//             for (int c = 0; c < OUTPUT_C; c++) {
//                 // Safely access the output array
//                 int index = c * OUTPUT_H * OUTPUT_W + h * OUTPUT_W + w;
//                 if (index < size) {
//                     float prob = output[index];
//                     if (prob > max_prob) {
//                         max_prob = prob;
//                         max_class = c;
//                     }
//                 }
//             }
            
//             // Assign color based on class
//             result.at<cv::Vec3b>(h, w) = colors[max_class];
//         }
//     }
    
//     return result;
// } 