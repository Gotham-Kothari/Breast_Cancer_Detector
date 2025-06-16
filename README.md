# 🧬 Breast Cancer Detection App

A deployed web application built with **Streamlit**, powered by an **SVM machine learning model** trained on the Breast Cancer Wisconsin dataset. This app predicts whether a tumor is **malignant** or **benign** based on 30 clinical features and securely logs prediction data using **Firebase Firestore**.

---

## 🚀 Live Demo

👉 [Launch the App]([https://YOUR-STREAMLIT-APP-URL](https://breastcancerdetector-zyusiavbnydvekapmguvrh.streamlit.app/))  

---

## 🔍 Features

- 📋 Input 30 diagnostic features from breast cancer cell nuclei
- 🧠 Predict tumor classification: **Malignant** or **Benign**
- 📈 PCA-based 2D visualization with diagnosis clusters
- 📊 Display model confidence score (via `predict_proba`)
- 🔐 Securely log inputs & predictions to **Firestore**
- ☁️ Firebase integration handled via **Streamlit Secrets**

---

## 📦 Tech Stack

| Component        | Tool/Library             |
|------------------|--------------------------|
| Frontend UI      | Streamlit                |
| ML Model         | scikit-learn (SVM + PCA) |
| Visualization    | Plotly Express           |
| Data Storage     | Firebase Firestore       |
| Hosting          | Streamlit Cloud          |
