# NASA-ML-exoplanet-identifier-model-Kepler-exo-detect-
Kepler Exo-Detect is an AI model that automates exoplanet discovery using NASA Exoplanet Archive data. It learns from confirmed planets and false positives to identify real signals, analyzing key astrophysical features. Built with Scikit-learn, it’s transparent, adaptive, and offers real-time visualization.

# Kepler ExoDetect

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Pipeline-orange?logo=scikitlearn)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Last Update](https://img.shields.io/badge/Last%20Updated-October%202025-lightgrey)

---

**Kepler ExoDetect** is a Streamlit-based machine learning application that automates the identification of exoplanet candidates using real data from the **NASA Exoplanet Archive (KOI cumulative table)**.  
It provides a transparent, research-oriented interface that enables training, testing, and visualization of models with clear interpretability.

---

## 🧩 Overview

Modern astronomy missions like **Kepler**, **TESS**, and **JWST** produce massive datasets where a large portion of detected signals are false positives. Manual analysis is slow and inconsistent.

**Kepler ExoDetect** uses AI to streamline this process by analyzing astrophysical parameters such as orbital period, stellar temperature, and transit depth to classify whether a signal represents a true exoplanet.  
It offers:
- Automated NASA data import and preprocessing  
- Real-time model training and validation  
- Probability-based predictions (not binary labels)  
- Interactive visualization and similarity search  
- Exportable trained pipelines for reuse  

---

## 🛰️ System Architecture



## ⚙️ Architecture and Workflow

### Data Flow

┌──────────────────────────────┐
│ NASA Exoplanet Archive (KOI) │
└──────────────┬───────────────┘
│
▼
┌────────────────────┐
│ Data Preprocessing │
│ Cleaning + Scaling │
└────────┬───────────┘
│
▼
┌────────────────────────────┐
│ Feature Engineering │
│ Select astrophysical vars │
└────────┬───────────────────┘
│
▼
┌──────────────────────────────┐
│ ML Pipeline (Random Forest) │
│ Scikit-learn + Cross-val │
└─────────┬────────────────────┘
│
▼
┌────────────────────────┐
│ Model Evaluation │
│ Metrics + Importance │
└────────┬───────────────┘
│
▼
┌────────────────────────┐
│ Streamlit Interface │
│ Predict + Visualize │
└────────────────────────┘


---

## ⚙️ Workflow Summary

NASA KOI Data → Preprocessing → Feature Engineering → ML Pipeline
→ Model Training → Evaluation → Streamlit Visualization → Prediction + Export


Model Details
Input Features
Feature	Description	Unit
pl_orbper	Orbital period	days
pl_rade	Planet radius	Earth radii
pl_trandep	Transit depth	ppm
pl_trandur	Transit duration	hours
st_teff	Stellar effective temperature	Kelvin
st_rad	Stellar radius	Solar radii
st_mass	Stellar mass	Solar masses

Algorithm

Random Forest Classifier (Scikit-learn)

Encapsulated in a preprocessing pipeline with StandardScaler

Adjustable hyperparameters from UI

Supports class balancing or weighted training

Evaluation Metrics
Metric	Description
Accuracy	Overall correctness of predictions
Precision	True exoplanets among predicted exoplanets
Recall	True exoplanets detected among all real ones
F1 Score	Balance between precision and recall
ROC-AUC	Separability between classes

Example Performance:

Model	Accuracy	F1	ROC-AUC
Random Forest	0.90	0.88	0.93

Streamlit Interface

Panels:

Data Panel: Load, clean, and preview KOI dataset

Training Panel: Configure model settings, balance data, and train

Evaluation Panel: View confusion matrix and metrics

Prediction Panel: Input custom features and get confidence-based predictions

Similarity Panel: Find nearest known exoplanets via normalized Euclidean distance

Model Export: Save trained models as .pkl files

Visualizations:

. Confusion Matrix
. Feature Importance Plot
. Probability Distribution
. SHAP-based interpretability (optional)

## Running the App
1. Clone the Repository
```python
git clone https://github.com/yourusername/Kepler-Exo-Detect.git
cd Kepler-Exo-Detect


2. Install Dependencies
pip install -r requirements.txt

3. Run the Application
streamlit run exoplanet_ml_streamlit_enhanced_v2.py
```

## 📦 Model Export and Reuse

Load and Use in Python
```python
import pickle

with open('exodetect_pipeline.pkl', 'rb') as f:
    data = pickle.load(f)

pipeline = data['pipeline']
features = data['feature_order']

sample = [365.2, 1.2, 400, 10.5, 5800, 1.0, 1.0]
print(pipeline.predict_proba([sample]))
```

📊 Example Prediction Output
```yaml
Prediction: Likely Exoplanet ✅
Confidence: 94.2%
Nearest Similar KOIs:
- KOI-142.01 (Similarity: 0.91)
- KOI-701.03 (Similarity: 0.88)
```

## Technical Summary
Data Preprocessing

NASA API fetch or fallback to koi_cumulative_backup.csv
Drop rows with missing astrophysical parameters
Normalize features using StandardScaler
Option to upsample minority class

Model Training
Random Forest (n_estimators=200, max_depth=15, min_samples_split=4)
5-fold stratified cross-validation
Accuracy, F1, ROC-AUC computed automatically

Similarity Search

Uses Euclidean distance between scaled feature vectors
Returns top 5 most similar exoplanets with similarity scores
Interpretability (Optional)
SHAP values for global and local feature importance
Explains contribution of each feature to a prediction

File Structure
```bash
Kepler-Exo-Detect/
│
├── exoplanet_ml_streamlit_enhanced_v2.py   # Main app
├── koi_cumulative_backup.csv               # Offline dataset
├── requirements.txt                        # Dependencies
└── README.md                               # Documentation
```

Requirements
```text
streamlit>=1.20
pandas>=1.4
numpy>=1.22
scikit-learn>=1.1
plotly>=5.6
requests>=2.28
shap>=0.41  # optional
```
```bash
pip install -r requirements.txt
```

















