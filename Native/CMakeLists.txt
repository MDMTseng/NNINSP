cmake_minimum_required(VERSION 3.18)
project(plating_defect_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add debug build type
set(CMAKE_BUILD_TYPE Debug)

# 設置平台特定的編譯選項
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")  # Add debug flags
    
    # 在 M1 Mac 上只使用 arm64 架構
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    else()
        set(CMAKE_OSX_ARCHITECTURES "x86_64")
    endif()
    
    # 設置 OpenCV 路徑（macOS with Homebrew）
    set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
    
    # macOS 使用 CoreML
    find_library(COREML CoreML REQUIRED)
    find_library(FOUNDATION Foundation REQUIRED)
    find_library(COREGRAPHICS CoreGraphics REQUIRED)
    find_library(COREIMAGE CoreImage REQUIRED)
    find_library(COREVIDEO CoreVideo REQUIRED)
    find_library(VISION Vision REQUIRED)
else(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")  # Add debug flags for MSVC
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    
    # Windows 需要 CUDA 和 TensorRT
    find_package(CUDA REQUIRED)
    find_package(TensorRT REQUIRED)
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 輸出調試信息
message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_OSX_ARCHITECTURES = ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")

# 確保 OpenCV 的包含路徑正確
include_directories(${OpenCV_INCLUDE_DIRS})

if(APPLE)
    # macOS 特定的源文件
    add_executable(plating_detector 
        src/main.cpp
        src/coreml_engine.mm
        src/coreml_engine_factory.mm
        src/image_processor.cpp
        src/inference_engine.cpp
        src/model_info.cpp
    )
    
    # 為 Objective-C++ 文件啟用 ARC
    set_source_files_properties(
        src/coreml_engine.mm
        PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
    
    target_link_libraries(plating_detector PRIVATE 
        ${OpenCV_LIBS}
        ${COREML}
        ${FOUNDATION}
        ${COREGRAPHICS}
        ${COREIMAGE}
        ${COREVIDEO}
        ${VISION}
    )
else()
    # Windows/Linux 特定的源文件
    add_executable(plating_detector 
        src/main.cpp
        src/tensorrt_engine.cpp
        src/image_processor.cpp
        src/inference_engine.cpp
    )
    
    target_link_libraries(plating_detector PRIVATE 
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${TensorRT_LIBRARIES}
        nvinfer
        nvonnxparser
    )
endif()

# 包含目錄
target_include_directories(plating_detector PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 設置 RPATH (for macOS)
if(APPLE)
    set_target_properties(plating_detector PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@executable_path;@executable_path/../lib"
    )
endif()

# Add ImageIO framework
find_library(IMAGEIO_LIBRARY ImageIO REQUIRED)
target_link_libraries(plating_detector PRIVATE ${IMAGEIO_LIBRARY})

# Optional: Silence deprecation warnings
target_compile_options(plating_detector PRIVATE -Wno-deprecated-declarations) 