import Foundation
import CoreML

let args = CommandLine.arguments
if args.count < 2 {
    print("Usage: swift test_model.swift <path_to_model>")
    exit(1)
}

let modelPath = args[1]
print("Testing model at path: \(modelPath)")

do {
    let modelURL = URL(fileURLWithPath: modelPath)
    print("Model URL: \(modelURL)")
    
    // Try to compile the model
    print("Attempting to compile model...")
    let compiledURL = try MLModel.compileModel(at: modelURL)
    print("Model compiled successfully to: \(compiledURL)")
    
    // Try to load the compiled model
    print("Attempting to load compiled model...")
    let model = try MLModel(contentsOf: compiledURL)
    print("Model loaded successfully!")
    
    // Print model metadata
    print("Model description: \(model.modelDescription)")
    if let metadata = model.modelDescription.metadata {
        print("Model metadata: \(metadata)")
    }
    
    print("Test completed successfully!")
} catch {
    print("Error: \(error)")
} 