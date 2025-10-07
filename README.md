# 🌱 Plant Disease detection  using VGG16

This project uses **Transfer Learning with the VGG16 model** to classify plant leaf images into 38 different plant disease categories. It is trained using a labeled and augmented dataset from Kaggle and implemented in **TensorFlow and Keras** on **Google Colab**.

---

## 📂 Dataset

- **Source**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: 38 plant diseases
- **Data Type**: Augmented images in `train/` and `valid/` folders
- **Input Size**: Resized to `224 x 224` for VGG16 compatibility

---

## 🧠 Model: VGG16 + Custom Classifier

The model uses VGG16 as the feature extractor (without the top layer), and a custom classification head is added to it.

### 🔧 Architecture:
- VGG16 base (frozen during training)
- `GlobalAveragePooling2D`
- `Dense(256, activation='relu')`
- `Dropout(0.5)`
- `Dense(38, activation='softmax')` (for 38 classes)

---

## 📊 Training

- **Platform**: Google Colab
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 10 (can be increased)
- **Metrics**: Accuracy

The model was evaluated on the validation set and showed high accuracy and good generalization.

---

## 📈 Results

- Training & validation accuracy plotted using `matplotlib`
- Classification report & confusion matrix visualized using `seaborn`

### Example Confusion Matrix:
![Confusion Matrix](confusion_matrix.png)

### Sample Prediction:
![Sample Prediction](sample_prediction.png)

---


