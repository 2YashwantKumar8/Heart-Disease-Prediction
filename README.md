# 🫀 Heart Disease Prediction Web App

A full-stack Flask web application that predicts the likelihood of heart disease based on a patient’s medical data using a machine learning model (XGBoost).
The app is trained on the UCI Heart Disease dataset and features a modern dark-themed UI for an engaging user experience.

---

## 🚀 Features

✅ Interactive web form with dark UI
✅ Real-time predictions using a trained XGBoost model
✅ Input validation and error handling
✅ Optimized and cleaned dataset
✅ Hyperparameter-tuned model
✅ Smooth animations and user-friendly interface
✅ Portable — runs locally (no need for cloud)

---

## 🎯 Purpose of the Project

Cardiovascular diseases are one of the leading causes of death worldwide.
Early diagnosis and intervention can save lives.

This project aims to:

* Provide an easy-to-use web interface for heart disease prediction
* Demonstrate how machine learning can be applied to healthcare data
* Showcase full-cycle ML deployment (data cleaning → model training → web app)

---

## 📊 Dataset Info

* Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
* Features:

  * `age`: Age in years
  * `sex`: Gender (0 = Female, 1 = Male)
  * `cp`: Chest pain type
  * `trestbps`: Resting blood pressure (mm Hg)
  * `chol`: Serum cholesterol (mg/dl)
  * `fbs`: Fasting blood sugar > 120 mg/dl
  * `restecg`: Resting electrocardiographic results
  * `thalach`: Maximum heart rate achieved
  * `exang`: Exercise induced angina
  * `oldpeak`: ST depression induced by exercise
  * `slope`: Slope of the peak exercise ST segment
  * `ca`: Number of major vessels colored by fluoroscopy
  * `thal`: Thalassemia
  * `num`: Target (0 = no heart disease, 1 = heart disease)

---

## 🛠️ Tech Stack

* Python 3.x
* Flask
* XGBoost
* Scikit-learn
* HTML5, CSS3 (dark theme)
* Jinja2 templates

---

## 📈 Model Details

* Model: XGBoost Classifier
* Hyperparameter tuning performed (GridSearchCV)
* Achieved accuracy: \~83% on test set
* Saved model: `xgboost_heart_model.pkl`

---
