# **CAGE-Assisted Semi-Supervised Object Classification**

## **Overview**
This project leverages the power of CAGE (Continuous and Quality-Guided Labeling Functions) and SPEAR (Semi-Supervised Data Programming) to address the challenges of object detection and classification in scenarios with limited labeled data. By utilizing unsupervised learning techniques, we enable models to detect and classify objects using pseudo-labels generated from multiple pre-trained models.

## **Features**
- **Multi-Model Pseudo Labeling**: Combines predictions from models like ResNet, VGG-16, and Inception to generate robust labels.
- **Joint Learning**: Utilizes SPEAR to jointly optimize labeling functions and object detection models.
- **Scalability**: Trains Faster R-CNN on unlabeled data, reducing dependency on large annotated datasets.
- **Explainability**: Incorporates explainable AI (GNNShap) for transparent training and predictions.

## **Problem Statement**
Traditional object detection heavily relies on labeled data, which is expensive and often unavailable. This project aims to determine how unlabeled data can be effectively utilized to train robust object detection models, ensuring scalability and accuracy.

## **Methodology**
1. **CAGE Labeling Functions**:
   - Continuous and discrete labeling functions derived from CNNs like ResNet and VGG-16.
   - Probabilistic scores for confidence and class predictions.
2. **Training Pipeline**:
   - Generate pseudo-labels for unlabeled data.
   - Train Faster R-CNN using SPEAR to jointly optimize labeling functions and model parameters.
   - Employ 100-epoch training with the Adam optimizer on the MSCOCO dataset.

## **Results**
- Achieved high individual accuracies (>90%) for ResNet, VGG-16, and Inception models in pseudo-label generation.
- Challenges remain in optimizing loss convergence and achieving consistent accuracy in joint learning frameworks.

## **References**
1. Chatterjee, O., Ramakrishnan, G., & Sarawagi, S. (2019). Data Programming using Continuous and Quality-Guided Labeling Functions. *ArXiv*, abs/1911.09860.
2. Abhishek, G.S., Ingole, H., Laturia, P., Dorna, V., Maheshwari, A., Ramakrishnan, G., & Iyer, R.K. (2021). SPEAR: Semi-supervised Data Programming in Python. *ArXiv*, abs/2108.00373.
