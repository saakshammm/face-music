# 🎭 Real-Time Facial Emotion Detection Using Deep Learning

This project detects human emotions in real-time using facial expressions captured through a webcam.  
It was built entirely with **TensorFlow/Keras, OpenCV, and Streamlit**, trained on the **Two datasets**, and deploys a **custom CNN model** for classification — not transfer learning.

#### Datasets:
1. https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
2. https://www.kaggle.com/datasets/msambare/fer2013
---

## 🧠 Overview

Facial Emotion Recognition helps computers understand human emotions by analyzing facial patterns.  
Our model classifies six emotions:

> **Angry | Fear | Happy | Neutral | Sad | Surprise**

The system detects faces from a live webcam feed, preprocesses them into 48×48 grayscale images, and predicts the emotion in real-time using a CNN model trained from scratch.

---

## 🧩 Key Features

- Custom **Convolutional Neural Network (CNN)** — no pre-trained model. (grayscale, 48×48 pixels) 
- Real-time **face detection** with OpenCV.  
- **Streamlit frontend** with live webcam display.  
- Evaluation metrics — accuracy, loss curves, confusion matrix, classification report.  
- Lightweight and completely local — no internet APIs required.

---

## 🗂️ Folder Structure
Emotion-Detection/
│
├── data
│ ├── train
│ ├── test
│
├── model/ 
│ ├── emotion_model.h5
│
├── confusion_matrix.png
├── training_plot.png
│
├── scripts/
│ ├── train_model.py 
│
├── main.py 
├── requirements.txt 
└── README.md
 
---

## ⚙️ Setup Instructions

### 1. Install Requirements
```bash
pip install -r requirements.txt
```
### 2. Train the Model
```bash
python scripts/train_model.py
```
#### This script:
Builds a CNN model (Conv2D, MaxPooling2D, Dense, Dropout)
Saves the trained model as emotion_model.h5
#### Generates:
- training_plot.png – accuracy & loss curves
- confusion_matrix.png – class performance visualization

### 3. Run the Real-Time App
```
streamlit run main.py
```
Then open the local URL shown in your terminal (usually http://localhost:8501).

## 🧮 Model Architecture
| Layer                     | Output Shape | Description                         |
| ------------------------- | ------------ | ----------------------------------- |
| Conv2D (32 filters, 3×3)  | 46×46×32     | Detects edges and corners           |
| MaxPooling2D (2×2)        | 23×23×32     | Reduces spatial size                |
| Conv2D (64 filters, 3×3)  | 21×21×64     | Detects facial parts                |
| MaxPooling2D (2×2)        | 10×10×64     | Keeps key patterns                  |
| Conv2D (128 filters, 3×3) | 8×8×128      | Learns full expressions             |
| MaxPooling2D (2×2)        | 4×4×128      | Compresses patterns                 |
| Flatten                   | —            | Converts to 1D                      |
| Dense (128) + ReLU        | —            | Learns emotion patterns             |
| Dropout (0.5)             | —            | Prevents overfitting                |
| Dense (6, Softmax)        | —            | Output probabilities for 6 emotions |

Optimizer: Adam (lr=1e-3)
Loss: Categorical Crossentropy
Metrics: Accuracy

## 📈 Training Results
Accuracy curves and confusion matrix are saved under /model/
Example outputs:
- training_plot.png → model accuracy & loss over epochs
- confusion_matrix.png → visual of predicted vs true labels

## 💻 Real-Time Detection (main.py)
- Uses OpenCV to capture webcam video
- Detects faces using Haarcascade classifier
- Crops and preprocesses each face (48×48 grayscale)
- Predicts emotion via trained CNN
- Displays emotion label and confidence on-screen
- Streamlit provides a dark, minimal UI for demo purposes

## 🧾 Tools & Libraries
| Category              | Tools Used          |
| --------------------- | ------------------- |
| Programming Language  | Python 3.10+        |
| Deep Learning         | TensorFlow / Keras  |
| Image Processing      | OpenCV              |
| Visualization         | Matplotlib, Seaborn |
| Frontend / Deployment | Streamlit           |
| Evaluation            | scikit-learn        |

## 🧱 System Requirements
| Component | Minimum                        | Recommended           |
| --------- | ------------------------------ | --------------------- |
| CPU       | Intel i5 / AMD equivalent      | i7+                   |
| GPU       | —                              | NVIDIA (CUDA 2–4 GB+) |
| RAM       | 8 GB                           | 16 GB                 |
| OS        | Windows 10/11, Linux, or macOS | —                     |
| Python    | 3.7+                           | 3.10+                 |

## 🚀 Future Enhancements
- Integrate with transfer learning (e.g., MobileNetV2) for higher accuracy
- Add multi-face detection and batch prediction
- Build dashboard view to analyze emotions over time
- Explore multimodal emotion recognition (audio + facial)

## 📚 References
1. OpenCV – Face detection using Haarcascade
2. TensorFlow/Keras Docs – CNN implementation examples
3. Analytics Vidhya / Medium – Emotion Recognition tutorials

## 👤 Authors
Saksham Kumar
Shreeya Barahpuriya
Department of Computer Applications — BCA V Semester

|| “By enabling computers to understand non-verbal cues, this system enhances human-computer interaction and contributes toward more adaptive, intelligent AI systems.” ||
