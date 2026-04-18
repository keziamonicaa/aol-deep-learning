# aol-deep-learning

🧬 Comparing MobileNetV3 vs GhostNet for Acute Lymphoblastic Leukemia Classification

📌 Project Overview
This repository presents a comparative study of MobileNetV3 and GhostNet, two lightweight Convolutional Neural Network (CNN) architectures, for the classification of Acute Lymphoblastic Leukemia (ALL) cells from microscopic blood smear images.

The motivation behind this work is to explore trade-offs between accuracy and computational efficiency. While complex models (ResNet, Vision Transformers, Hybrid CNNs) achieve very high accuracy (>97%), they are computationally expensive and difficult to deploy in rural or resource-limited healthcare settings. Lightweight models such as MobileNetV3 and GhostNet offer a balance between performance and efficiency, making them suitable for mobile and edge deployment.

📊 Dataset Description
> Source: Taleqani Hospital, Tehran (Iran) – Bone marrow laboratory

> Size: 3,256 Peripheral Blood Smear (PBS) images from 89 suspected ALL patients

> Preparation: Blood samples stained by skilled laboratory staff, imaged using Zeiss camera at 100x magnification, saved in JPG format

> Segmentation: HSV color thresholding applied to provide segmented images

Classes:
> Benign (hematogones)

> Early Pre-B ALL

> Pre-B ALL

> Pro-B ALL

Initial Distribution:

> Benign: 504

> Early: 985

> Pre: 963

> Pro: 804
Balanced Dataset (after augmentation): 1,500 images per class

⚙️ Preprocessing Pipeline
> Resizing: All images resized to (128 × 128) pixels

> Normalization: Pixel values scaled to [0, 1]

> Augmentation:
  Rotation (up to 90°)
  Horizontal & vertical flips
  Width & height shifts
  Zoom (±20%)
  Brightness adjustments (0.8–1.2)
  Shear transformations
  Ensures balanced dataset and prevents overfitting

🏗️ Baseline CNN
> Architecture: 5 convolutional blocks (Conv2D → BatchNorm → MaxPooling → Dropout)

> Filters: 16 → 32 → 64 → 128 → 256

> Dense Layers: Two fully connected layers with ReLU activation, followed by Softmax for 4-class classification

> Optimizer: Adam (lr = 1e-4)

> Loss Function: Categorical Crossentropy

> Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

> Performance (Validation): Accuracy ~79%

Strong performance on “Pre” class
Weak recall on “Benign” due to class imbalance

🔬 MobileNetV3 (Transfer Learning)
> Base Model: MobileNetV3Large (pre-trained on ImageNet, include_top=False)

> Custom Head:
GlobalAveragePooling2D
Dense(128, ReLU)
Dense(4, Softmax)

> Evaluation: 5-fold stratified cross-validation for robust performance estimation

> Training: 30 epochs, Adam optimizer (lr = 1e-4)

Optional Fine-Tuning: Unfreeze base layers, retrain with lr = 1e-5

Advantages:
Lightweight and efficient, Suitable for mobile deployment, Strong generalization with transfer learning

🔬 GhostNet (Transfer Learning)
> Base Model: GhostNet (pre-trained on ImageNet, include_top=False)

> Custom Head: Same as MobileNetV3 (GlobalAveragePooling2D → Dense → Softmax)

> Evaluation: 5-fold stratified cross-validation

> Training: 30 epochs, Adam optimizer (lr = 1e-4)

Optional Fine-Tuning: Unfreeze base layers, retrain with lr = 1e-5

> Advantages:
Designed for extreme efficiency (ghost modules reduce redundancy in feature maps)
Comparable accuracy to MobileNetV3

Ideal for edge devices and low-power environments

🖥️ Training Environment
Framework: TensorFlow / Keras
Hardware: NVIDIA T4 GPU (Google Colab)
Visualization: TensorBoard for monitoring loss/accuracy curves
