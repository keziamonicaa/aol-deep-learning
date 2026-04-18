# aol-deep-learning
Comparing MobileNetV3 vs GhostNet For Acute Lymphoblastic Leukemia Classification

📌 Overview
This project explores the application of lightweight CNN architectures (MobileNetV3 and GhostNet) for classifying Acute Lymphoblastic Leukemia (ALL) cells from microscopic images. The goal is to evaluate trade-offs between accuracy and computational efficiency, making AI-assisted diagnosis more accessible in rural and resource-limited healthcare settings.

📊 Dataset
Source: Taleqani Hospital, Tehran (Iran) – Bone marrow laboratory

Size: 3,256 Peripheral Blood Smear (PBS) images from 89 suspected ALL patients

Classes:

Benign (hematogones)

Early Pre-B ALL

Pre-B ALL

Pro-B ALL

Image Specs: Zeiss camera, 100x magnification, JPG format

Segmentation: HSV color thresholding provided

Class Distribution (before augmentation):

Benign: 504

Early: 985

Pre: 963

Pro: 804

Balanced Dataset (after augmentation): 1,500 images per class

⚙️ Preprocessing
Resizing: All images resized to (128 × 128)

Normalization: Pixel values scaled to [0, 1]

Augmentation:

Rotation, flips, zoom, brightness, shear, shifts

Ensures balanced dataset and prevents overfitting

🏗️ Baseline CNN
Architecture: 5 convolutional blocks (Conv2D → BatchNorm → MaxPooling → Dropout)

Optimizer: Adam (lr = 1e-4)

Loss: Categorical Crossentropy

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

Performance (Validation): Accuracy ~79%, with class imbalance challenges

🔬 MobileNetV3 (Transfer Learning)
Base Model: MobileNetV3Large (pre-trained on ImageNet, include_top=False)

Custom Head: GlobalAveragePooling2D → Dense(128, ReLU) → Dense(4, Softmax)

Evaluation: 5-fold stratified cross-validation

Training: 30 epochs, Adam optimizer (lr = 1e-4)

Optional Fine-Tuning: Unfreeze base layers, retrain with lr = 1e-5

Metrics: Accuracy, Precision, Recall, F1-score per class

🔬 GhostNet (Transfer Learning)
Base Model: GhostNet (pre-trained on ImageNet, include_top=False)

Custom Head: Same as MobileNetV3 (GlobalAveragePooling2D → Dense → Softmax)

Evaluation: 5-fold stratified cross-validation

Training: 30 epochs, Adam optimizer (lr = 1e-4)

Optional Fine-Tuning: Unfreeze base layers, retrain with lr = 1e-5

Metrics: Accuracy, Precision, Recall, F1-score per class

🖥️ Training Environment
Framework: TensorFlow / Keras

Hardware: NVIDIA T4 GPU (Google Colab)

Visualization: TensorBoard for monitoring loss/accuracy curves

📈 Results & Analysis
Baseline CNN: Moderate accuracy (~79%), struggled with class imbalance

MobileNetV3: Improved performance with transfer learning and cross-validation

GhostNet: Comparable lightweight alternative, designed for efficiency in mobile/edge deployment

Trade-Offs:

Complex models (ViT, Hybrid CNNs) achieve >97% accuracy but require heavy computation

Lightweight models (MobileNetV3, GhostNet) balance accuracy with deployment feasibility in rural areas
