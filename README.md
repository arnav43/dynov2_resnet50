# Skin Cancer Detection: Comparing CNN (ResNet-50) and Transformer (DINOv2) Architectures

Overview
This repository contains two models, DINOv2 (a Vision Transformer-based architecture) and ResNet-50 (a Convolutional Neural Network), trained using transfer learning on a skin cancer detection dataset. The main goal of this project is to compare Transformer-based architectures with CNNs in terms of performance (accuracy, latency, etc.) under similar conditions.

Project Details
Dataset: Skin cancer dataset
Task: Binary classification (Melanoma detection)
Approach: Transfer learning using two different backbones:
ResNet-50 (CNN-based)
DINOv2 (Transformer-based)
Training Setup
Backbone Frozen: For both models, the backbone was kept frozen, and only the final fully connected (linear) layers were unfrozen.
Trainable Parameters: About 4k trainable parameters in the final linear layers for both models.
Epochs: 10 epochs.
Batch Size: The same batch size was used for both models.
Learning Rate: Identical learning rates for both models.
Optimizers: Same optimizer (e.g., Adam or SGD) for both models.
Loss Function: Binary cross-entropy loss function was used.
Data Preprocessing
The following transformations were applied to the input images:

Training Data:
Resize to (224, 224)
Random Horizontal and Vertical Flip (p=0.5)
Random Rotation (15 degrees)
Tensor conversion and normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Testing Data:
Resize to (224, 224)
Tensor conversion and normalization (same as above)
Files Included
DINOv2 Model: dinov2_finetuned.pth
ResNet-50 Model: resnet50_finetuned.pth
Training Notebooks:
resnet50_training.ipynb
dinov2_training.ipynb
Results
Goal: To evaluate and compare the performance of the Transformer-based architecture (DINOv2) and the CNN-based architecture (ResNet-50) in terms of accuracy, speed (latency), and GPU memory usage.
Usage
To run the models or replicate the training process, follow the steps provided in the notebooks.
