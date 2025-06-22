# 🎯 RNN Gesture Recognition System

A comprehensive Computer Vision and Machine Learning project for gesture recognition using Recurrent Neural Networks (RNNs) with advanced video processing capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Video Processing Tools](#video-processing-tools)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a sophisticated gesture recognition system using **ConvLSTM2D** networks to classify human gestures from video sequences. The system processes videos by extracting frames, converting them to binary representations, and using Connected Component Analysis (CCA) for feature extraction before feeding them into RNN models for classification.

### Supported Gestures
- **FistVert** - Vertical fist gesture
- **HeadTurn** - Head turning movements
- **HeadLeft** - Head movements to the left
- **HeadRight** - Head movements to the right

## ✨ Features

### 🧠 Machine Learning
- **ConvLSTM2D** architecture for spatiotemporal feature learning
- Binary image processing with threshold-based conversion
- Connected Component Analysis (CCA) for motion detection
- Frame subtraction for motion extraction
- Multiple model variations with regularization support
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score, etc.)

### 🎥 Video Processing
- Real-time camera capture with **Kivy** interface
- Frame extraction from video files
- Video mirroring and time reversal
- Automatic video preprocessing and augmentation
- Batch video processing capabilities

### 🛠️ Utilities
- File management and renaming tools
- Binary image conversion utilities
- Data preprocessing and augmentation pipeline

## 🏗️ System Architecture

```
Video Input → Frame Extraction → Frame Subtraction → Binary Conversion → 
CCA Processing → Feature Extraction → ConvLSTM2D → Gesture Classification
```

### Processing Pipeline
1. **Video Capture**: Extract frames at specified frame rates
2. **Frame Subtraction**: Calculate differences between consecutive frames
3. **Binary Conversion**: Apply threshold to create binary images
4. **Connected Components**: Identify and extract motion regions
5. **RNN Processing**: Feed processed sequences to ConvLSTM2D model
6. **Classification**: Output gesture predictions with confidence scores

## 🚀 Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, CPU mode available)

### Dependencies Installation

```bash
pip install tensorflow
pip install keras
pip install opencv-python
pip install scikit-learn
pip install numpy
pip install matplotlib
pip install moviepy
pip install kivy
```

### Alternative Installation
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Training a Model

1. **Prepare your dataset** in the following structure:
```
data/
├── FistVert/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── HeadTurn/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
```

2. **Configure the data path** in the training script:
```python
data_dir = "path/to/your/data"
```

3. **Run training**:
```bash
python Binary_RNN.py
```

### Testing the Model

```bash
python Rnn_test.py
```

### Real-time Camera Capture

```bash
python camera.py
```

### Video Processing Tools

**Extract frames from video:**
```bash
python ExtractFrames.py
```

**Mirror videos:**
```bash
python Mirror_Videos.py
```

**Reverse videos:**
```bash
python DupReversedVideo.py
```

## 📁 Project Structure

```
├── Binary_RNN.py              # Main binary RNN implementation
├── Rnn_Network.py             # Standard RNN network
├── rnn_binary_regulated.py    # Regularized RNN with advanced features
├── Rnn_test.py                # Model testing and evaluation
├── camera.py                  # Real-time camera interface
├── ExtractFrames.py           # Video frame extraction utility
├── ToBinary.py                # Binary image conversion
├── DupReversedVideo.py        # Video time reversal tool
├── Mirror_Videos.py           # Video mirroring utility
├── Change_File_Name.py        # Video file renaming
├── RenameFiles.py             # General file renaming utility
└── README.md                  # Project documentation
```

## 🤖 Models

### ConvLSTM2D Architecture

The system uses **ConvLSTM2D** layers that combine convolutional operations with LSTM capabilities:

- **Input Shape**: `(sequence_length, height, width, channels)`
- **Filters**: 64 (first layer), 32 (second layer)
- **Kernel Size**: 3x3
- **Regularization**: L2 regularization with dropout
- **Output**: Softmax/Sigmoid activation for multi-class classification

### Model Variations

1. **Binary_RNN.py**: Basic binary processing model
2. **Rnn_Network.py**: Standard RGB processing model  
3. **rnn_binary_regulated.py**: Advanced model with regularization

### Training Configuration

```python
# Configurable parameters
seq_len = 20                    # Sequence length
pixel_size = 64                 # Image resolution
frameRate = 2                   # Frame extraction rate
threshold_value = 30            # Binary threshold
batch_size = 4                  # Training batch size
epochs = 30                     # Training epochs
```

## 🎬 Video Processing Tools

### Frame Extraction
- Automatic frame extraction at specified intervals
- Configurable frame rates and resolution
- Support for various video formats

### Video Augmentation
- **Time Reversal**: Create reversed video sequences
- **Mirroring**: Horizontal mirroring for data augmentation
- **FPS Control**: Standardize frame rates across videos

### Real-time Capture
- **Kivy**-based camera interface
- Timestamped image capture
- Live preview with controls

## 🔧 Configuration

### Key Parameters

```python
# Model Configuration
seq_len = 20                    # Number of frames per sequence
classes = ["FistVert", "HeadTurn"]  # Gesture classes
pixel_size = 64                 # Image resolution (64x64)
frameRate = 2                   # Frame sampling rate
threshold_value = 30            # Binary conversion threshold

# Training Configuration  
batch_size = 4                  # Training batch size
epochs = 30                     # Training epochs
validation_split = 0.2          # Validation data percentage
test_size = 0.20               # Test data percentage
```

### Hardware Configuration

```python
# GPU Configuration (disable GPU if needed)
tf.config.experimental.set_visible_devices([], 'GPU')
```

## 📊 Performance Metrics

The system provides comprehensive evaluation metrics:
- **Accuracy Score**
- **Classification Report**  
- **Confusion Matrix**
- **Precision, Recall, F1-Score**
- **Cohen's Kappa Score**
- **ROC AUC Score**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 coding standards
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras** team for the deep learning framework
- **OpenCV** community for computer vision tools
- **Kivy** developers for the UI framework
- **MoviePy** contributors for video processing capabilities

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review existing issues and discussions

---

**Made with ❤️ for the Computer Vision and Machine Learning community** 