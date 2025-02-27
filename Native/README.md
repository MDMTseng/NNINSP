# 鋼板缺陷檢測工具

這個工具用於檢測鋼板表面缺陷，支持 Windows (NVIDIA GPU) 和 macOS (Apple Silicon/Intel) 平台。

## 系統要求

### Windows
- NVIDIA GPU
- CUDA Toolkit 11.x
- TensorRT 8.x
- Visual Studio 2019 或更新版本
- CMake 3.18+
- OpenCV 4.x
- Python 3.8+

### macOS
- Xcode 命令行工具
- CMake 3.18+
- OpenCV 4.x
- Python 3.8+

## 安裝步驟

### Windows

1. 安裝 Visual Studio 2019 或更新版本

2. 安裝 CUDA Toolkit：
   - 訪問 [NVIDIA CUDA 下載頁面](https://developer.nvidia.com/cuda-downloads)
   - 選擇對應版本下載並安裝

3. 安裝 TensorRT：
   - 訪問 [NVIDIA TensorRT 下載頁面](https://developer.nvidia.com/tensorrt)
   - 下載並安裝對應版本

4. 安裝 OpenCV：
   ```powershell
   # 使用 vcpkg
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   .\vcpkg install opencv4:x64-windows
   ```

5. 安裝 Python 依賴：
   ```powershell
   pip install onnx onnxruntime
   ```

### macOS

1. 安裝 Homebrew（如果還沒安裝）：
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. 安裝依賴：
   ```bash
   brew install cmake opencv python@3.8
   pip3 install torch torchvision onnx coremltools
   ```

## 編譯步驟

1. 轉換模型：
   ```bash
   # 在 Native 目錄下
   python convert_model.py
   ```
   這將生成 `.onnx` 文件，在 macOS 上會自動額外生成 `.mlmodel` 文件

2. 創建構建目錄：
   ```bash
   mkdir build
   cd build
   ```

3. 配置專案：

   Windows:
   ```powershell
   cmake -G "Visual Studio 16 2019" -A x64 ..
   ```

   macOS:
   ```bash
   cmake ..
   ```

4. 編譯：

   Windows:
   ```powershell
   cmake --build . --config Release
   ```

   macOS:
   ```bash
   make
   ```

## 使用方法

1. 準備模型和測試圖片

2. 運行程序：

   Windows:
   ```powershell
   .\Release\plating_detector.exe ..\model.onnx test_image.jpg
   ```

   macOS:
   ```bash
   ./plating_detector ../model.mlmodel test_image.jpg
   ```

## 輸出說明

程序會顯示一個視窗展示檢測結果，並將結果保存為 `result.png`。

顏色說明：
- 黑色：正常區域
- 紅色：缺陷類型 1
- 綠色：缺陷類型 2
- 藍色：缺陷類型 3
- 黃色：缺陷類型 4
- 青色：缺陷類型 5

## 常見問題

1. **找不到 CUDA**
   - 確保 CUDA Toolkit 已正確安裝
   - 檢查環境變量是否正確設置

2. **找不到 TensorRT**
   - 確保 TensorRT 已正確安裝
   - 檢查 CMake 中的 TensorRT 路徑是否正確

3. **macOS 上的編譯錯誤**
   - 確保已安裝 Xcode 命令行工具
   - 檢查 OpenCV 是否正確安裝

4. **運行時錯誤**
   - 檢查模型文件路徑是否正確
   - 確保輸入圖片格式正確（支持 jpg、png）

## 性能優化

- Windows：啟用了 TensorRT 的 FP16 優化（如果硬體支持）
- macOS：使用 Metal Performance Shaders 進行加速
- 首次運行時會進行模型優化，可能需要較長時間
- 後續運行會使用優化後的模型，速度會更快

## 貢獻指南

如果您想貢獻代碼，請：
1. Fork 本專案
2. 創建您的特性分支
3. 提交您的改動
4. 推送到您的分支
5. 創建 Pull Request

## 許可證

[MIT License](LICENSE) 