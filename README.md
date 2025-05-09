# Vehicle Classification with CNN + Attention
This project is the second assignment in a computer vision course and focuses on building a vehicle classification system using a CNN-based model enhanced with an attention mechanism. The dataset includes images of various car classes.

---

## Dataset Preparation Script (`preprocess_cars_data.ipynb`)

This script is responsible for preparing the car dataset before training the main model. It includes several key steps to ensure the dataset is clean, well-structured, and ready for use in training.

### Main Functionalities:
- **Downloading and extracting** the dataset from Kaggle using the Kaggle API.
- **Applying CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.
- **Filtering and detecting cars** using a pre-trained YOLO model and saving their bounding box annotations.
- **Cleaning the dataset** by removing images without any valid car detections.
- **Cropping car regions** from images based on the detected bounding boxes and organizing them into class-specific folders.

> This preprocessing pipeline ensures that only relevant and high-quality images are used in the training phase.


---

## Dataset Distribution

The number of images per class is visualized using a bar chart to show the class imbalance or balance. This is useful for understanding the training dynamics and potential need for data augmentation.

![classes](https://github.com/user-attachments/assets/ad998050-0032-4f0b-82d7-d30c5bee2473)

---

## Model Architecture: CNN + Attention

The core model is based on a **pre-trained ResNet-50** backbone with an additional **attention mechanism** to improve focus on informative regions within the feature maps.

### Components:

- **Feature Extractor**: ResNet-50 (ImageNet weights)
- **Attention Module**: Custom attention mechanism that re-weights spatial features
- **Classifier**: Fully connected layer with dropout

---


## Training Setup

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Dropout**: 0.2
- **Metrics Tracked**: Training/Validation Loss & Accuracy

### Training Visualization

- Training and validation loss are plotted over epochs
- Accuracy curves are also visualized
- These plots are helpful for diagnosing overfitting/underfitting


![train](https://github.com/user-attachments/assets/49ddb453-b1e1-4e3c-9e3b-b363e57861fc)


---

## Evaluation Metrics


- **Top-1 Accuracy**: Measures exact match
- **Top-5 Accuracy**: Checks if the correct class is among the top 5 predictions
- **Classification Report**: Includes precision, recall, and F1-score for each class
- **Confusion Matrix**: Heatmap to visualize classification performance

```
               precision    recall  f1-score   support

   Mazda-2000       0.75      0.90      0.82        20
Nissan-Zamiad       0.95      1.00      0.97        19
  Peugeot-206       0.63      0.80      0.71        15
 Peugeot-207i       0.62      0.87      0.72        15
  Peugeot-405       0.89      0.62      0.73        13
 Peugeot-Pars       0.85      0.73      0.79        15
       Peykan       0.80      0.62      0.70        13
    Pride-111       0.62      0.53      0.57        15
    Pride-131       0.73      0.79      0.76        24
         Quik       0.74      0.87      0.80        30
  Renault-L90       0.77      0.53      0.62        19
       Samand       1.00      0.76      0.86        21
        Tiba2       0.81      0.81      0.81        21

     accuracy                           0.77       240
    macro avg       0.78      0.76      0.76       240
 weighted avg       0.78      0.77      0.77       240

```

- **Top-1 Accuracy**: 0.7708
- **Top-5 Accuracy**: 0.9542


## Confusion Matrix
![confusion_matrix](https://github.com/user-attachments/assets/944cfc74-4d91-4779-8258-adababc8a6a1)



![test_model](https://github.com/user-attachments/assets/b88c30a0-48cc-4213-a498-cce6a279dc99)
