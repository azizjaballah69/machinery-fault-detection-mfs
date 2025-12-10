# Machinery Fault Detection - MFS Dataset

A complete machine learning pipeline for detecting machinery faults (imbalance conditions) using the UFRJ Machinery Fault Simulator (MFS) dataset with advanced signal processing and machine learning techniques.

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📋 Overview

This project implements a **complete end-to-end pipeline** for machinery fault detection using accelerometer data from the UFRJ Machinery Fault Simulator. The pipeline processes **12.25 million raw accelerometer readings** from 49 experimental files and trains a machine learning model achieving **98.47% test accuracy**.

### Key Capabilities:
- ✅ Loads and combines 49 CSV files automatically
- ✅ Extracts 40 statistical features (Mean, Std, Max, Min, RMS) from 8 sensors
- ✅ Detects anomalies using Isolation Forest
- ✅ Trains Random Forest classifier with supervised learning
- ✅ Generates comprehensive visualizations and reports
- ✅ Complete logging and results tracking

---

## 📊 Dataset

**Source:** [UFRJ Machinery Fault Simulator (MFS)](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html)

**Dataset Characteristics:**
- **Total Samples:** 12,250,000 accelerometer readings
- **Number of Files:** 49 CSV files
- **Sensors:** 8 accelerometers
- **Sampling Rate:** 10 kHz
- **Duration:** ~125 seconds per file
- **Fault Types:** 4 classes based on imbalance severity
  - **Class 0 (HEALTHY):** Imbalance value < 30g
  - **Class 1 (LIGHT IMBALANCE):** Imbalance 30-40g
  - **Class 2 (MODERATE IMBALANCE):** Imbalance 40-60g
  - **Class 3 (SEVERE IMBALANCE):** Imbalance > 60g

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/azizjaballah69/machinery-fault-detection-mfs.git
cd machinery-fault-detection-mfs
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Usage

**Run the complete pipeline:**
```bash
python src/complete_pipeline_mfs.py
```

This will automatically:
1. Load all 49 CSV files from your data folder
2. Combine and preprocess the data
3. Extract 40 features from 8 sensors
4. Detect anomalies (10% contamination)
5. Perform statistical analysis
6. Train Random Forest classifier
7. Generate visualizations and reports

---

## 📁 Project Structure

```
machinery-fault-detection-mfs/
├── src/
│   ├── complete_pipeline_mfs.py          # Main pipeline (all-in-one)
│   ├── load_data.py                      # Data loading module
│   ├── preprocess_6g.py                  # Feature extraction
│   ├── anomaly_detect.py                 # Anomaly detection
│   ├── statistical_analysis.py           # Statistical analysis
│   └── train_random_forest.py            # Model training
├── data/
│   └── README.md                         # Data folder info
├── results/
│   └── README.md                         # Results folder info
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Git ignore rules
├── LICENSE                               # MIT License
└── README.md                             # This file
```

---

## 📈 Pipeline Steps

### **Step 1: Data Loading**
- Loads 49 CSV files from the data folder
- Combines into single DataFrame (12.25M rows)
- Assigns fault labels based on filename values

### **Step 2: Feature Extraction**
- Creates 1024-sample sliding windows (50% overlap)
- Extracts 5 statistical features per sensor:
  - **Mean:** Average signal value
  - **Std Dev:** Standard deviation
  - **Max:** Maximum value
  - **Min:** Minimum value
  - **RMS:** Root mean square
- Total: 40 features (5 × 8 sensors)
- Result: 23,924 feature vectors

### **Step 3: Normalization**
- Applies StandardScaler to normalize all features
- Zero mean, unit variance

### **Step 4: Anomaly Detection**
- Uses Isolation Forest algorithm
- Contamination: 10% (2,393 anomalies detected)
- Generates anomaly scores for each sample

### **Step 5: Statistical Analysis**
- Calculates feature distributions
- Analyzes feature correlations
- Identifies important features by variance
- Saves statistical summaries

### **Step 6: Model Training**
- Creates 4 imbalance severity classes from true labels
- Splits: 70% train, 15% validation, 15% test
- Trains Random Forest Classifier (100 trees, depth=15)
- **Results:**
  - Train Accuracy: 100%
  - Validation Accuracy: 98.08%
  - Test Accuracy: **98.47%** ✅

### **Step 7: Visualization**
- Feature distributions by fault type
- Correlation heatmap
- Anomaly analysis charts
- Model performance metrics

---

## 📊 Results

### **Model Performance**
```
Training Summary:
├── Train Accuracy:       100.00%
├── Validation Accuracy:   98.08%
├── Test Accuracy:         98.47% ✅
├── Total Samples:        23,924
└── Total Features:            40

Fault Classification:
├── HEALTHY:           3,289 samples (13.7%)
├── LIGHT IMBALANCE:   8,235 samples (34.4%)
├── MODERATE IMBALANCE: 9,148 samples (38.2%)
└── SEVERE IMBALANCE:   3,252 samples (13.6%)

Anomalies Detected:
├── Total:             2,393 (10.0%)
└── Normal:           21,531 (90.0%)
```

### **Top 5 Most Important Features**
1. **col1_mean** (12.24%) - Sensor 1 mean value
2. **col4_mean** (11.26%) - Sensor 4 mean value
3. **col0_max** (10.47%) - Sensor 0 maximum value
4. **col3_max** (8.21%) - Sensor 3 maximum value
5. **col2_max** (7.90%) - Sensor 2 maximum value

---

## 📋 Output Files

### **Data Files Generated:**
- `combined_6g_data_with_labels.csv` - Combined raw data (12.25M rows)
- `processed_features_with_labels.csv` - Extracted 40 features (23,924 samples)
- `anomaly_detection_results_with_labels.csv` - Anomaly labels and scores
- `feature_statistics.csv` - Statistical summary

### **Model Files:**
- `random_forest_model_mfs.pkl` - Trained Random Forest model
- `model_predictions_mfs.csv` - Test set predictions

### **Visualization Charts:**
- `01_fault_distributions.png` - Feature distributions by fault type
- `02_correlation_heatmap.png` - Feature correlation matrix
- `03_anomaly_analysis.png` - Anomaly detection results
- `04_model_performance_mfs.png` - Model metrics and feature importance

### **Logging:**
- `pipeline_log_[timestamp].txt` - Complete execution log

---

## 🛠️ Dependencies

All dependencies are listed in `requirements.txt`:

```
pandas>=2.0.0          # Data manipulation
numpy>=1.23.0          # Numerical computing
scikit-learn>=1.2.0    # Machine learning
scipy>=1.10.0          # Scientific computing
matplotlib>=3.5.0      # Visualization
seaborn>=0.12.0        # Statistical visualization
joblib>=1.2.0          # Model persistence
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📝 Usage Examples

### **Run Complete Pipeline**
```bash
python src/complete_pipeline_mfs.py
```

### **Run Individual Steps**
```bash
# Load and combine data
python src/load_data.py

# Extract features
python src/preprocess_6g.py

# Detect anomalies
python src/anomaly_detect.py

# Statistical analysis
python src/statistical_analysis.py

# Train model
python src/train_random_forest.py
```

### **Use Pre-trained Model**
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('random_forest_model_mfs.pkl')

# Load features
features = pd.read_csv('processed_features_with_labels.csv')

# Make predictions
predictions = model.predict(features.iloc[:, :-2])
```

---

## 🔍 Key Features

### **Automated Data Processing**
- Automatically detects and loads all CSV files
- Handles large datasets efficiently
- Memory-optimized feature extraction

### **Comprehensive Fault Classification**
- 4-class imbalance severity classification
- True fault labels from MFS dataset
- Supervised learning approach

### **Advanced Analytics**
- Isolation Forest anomaly detection
- Correlation analysis
- Feature importance ranking

### **Professional Reporting**
- Detailed execution logging
- Multiple visualization charts
- Statistical summaries

---

## 🎓 Methodology

### **Signal Processing**
1. Sliding window segmentation (1024 samples, 50% overlap)
2. Feature extraction from each window
3. Normalization using StandardScaler

### **Machine Learning**
1. Supervised classification (4 imbalance classes)
2. Train-validation-test split (70-15-15)
3. Random Forest ensemble method
4. Hyperparameters: 100 trees, max_depth=15

### **Validation**
- Stratified split to maintain class distribution
- Cross-validation metrics (train/val/test)
- Confusion matrix analysis
- Classification reports

---

## 📚 References

- **Dataset:** UFRJ Machinery Fault Simulator (MFS)
  - URL: https://www02.smt.ufrj.br/~offshore/mfs/page_01.html
  
- **Machine Learning:**
  - Scikit-learn: https://scikit-learn.org/
  - Random Forest: Breiman, L. (2001)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Add new features

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License
Copyright (c) 2025 Aziz Jaballah
```

---

## 📧 Contact

- **GitHub:** [@azizjaballah69](https://github.com/azizjaballah69)
- **Project:** machinery-fault-detection-mfs

---

## 🙏 Acknowledgments

- UFRJ (Federal University of Rio de Janeiro) for the Machinery Fault Simulator dataset
- Scikit-learn and pandas teams for excellent ML libraries

---

**Last Updated:** December 10, 2025

**Status:** Active ✅ | **Maintained:** Yes | **License:** MIT