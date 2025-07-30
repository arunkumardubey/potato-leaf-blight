# potato-leaf-blight

# 🍃 Potato Leaf Disease Classification Using Deep Learning

## 📝 Description

This project implements a robust deep learning framework for classifying potato leaf diseases—**Early Blight**, **Late Blight**, and **Healthy**—using transfer learning and ensemble stacking techniques. The core idea is to benchmark various pretrained CNN architectures and enhance performance using a stacked ensemble of base learners with meta-classifiers.

## 📁 Dataset Information

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes Used:**
  - Early Blight: 1,000 images
  - Late Blight: 1,000 images
  - Healthy: 152 images (augmented to address class imbalance)

## 💻 Code Overview

- Implemented in a Jupyter Notebook: `potatooo.ipynb`
- Models evaluated: `VGG16`, `InceptionV3`, `EfficientNetB0`, `DenseNet121`
- Ensemble Strategy: Stacked ensemble using **softmax outputs** of 3 models with meta-classifiers:
  - Logistic Regression
  - Random Forest
  - AdaBoost

## 🧪 Usage Instructions

### Step 1: Clone Repository or Upload Files
```bash
git clone https://github.com/your-username/potato-leaf-blight-detection.git](https://github.com/arunkumardubey/potato-leaf-blight
cd potato-leaf-blight
```

### Step 2: Install Requirements
Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install tensorflow keras scikit-learn matplotlib numpy pandas opencv-python seaborn
```

### Step 3: Prepare Dataset
Download and extract the PlantVillage dataset:
- Navigate to the [Kaggle dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Organize data into three folders: `Early Blight`, `Late Blight`, `Healthy` under `data/` directory.

### Step 4: Run the Notebook
Launch Jupyter Notebook and run:
```bash
jupyter notebook potatooo.ipynb
```

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV

GPU support (e.g., NVIDIA CUDA) is recommended for faster training.

## 🧠 Methodology

1. **Data Balancing:** Augmented the underrepresented “Healthy” class using duplication-based augmentation.
2. **Preprocessing:** Resizing, normalization, augmentation (flip, zoom, brightness, rotation)
3. **Model Training:** Transfer learning with frozen base layers → fine-tuning using Adam optimizer
4. **Stacked Ensemble:** Concatenate softmax outputs and classify using meta-classifiers
5. **Evaluation:** Accuracy up to **98.3%**, visual explanations via Grad-CAM

## 📊 Results

- **Best Model:** Stacked Ensemble with Logistic Regression (98.3% accuracy)
- **Best Individual Model:** DenseNet121 (98.1%)
- **Improved generalization** through ensemble learning

## 📲 Deployment Notes

- Models can be deployed on edge/mobile devices using TensorFlow Lite or ONNX.
- Proposed mobile app for real-time potato leaf diagnosis in field conditions.


## 📄 License

This project is intended for academic and research purposes only. For commercial usage, please contact the authors.


