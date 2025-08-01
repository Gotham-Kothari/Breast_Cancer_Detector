# 🧬 Breast Cancer Detection App

A deployed web application built with **Streamlit**, powered by an **SVM machine learning model** trained on the Breast Cancer Wisconsin dataset. This app predicts whether a tumor is **malignant** or **benign** based on 30 clinical features and securely logs prediction data using **Firebase Firestore**.

---

## 🚀 Live Demo

👉 [Launch the App] (https://breastcancerdetector-zyusiavbnydvekapmguvrh.streamlit.app/)
![image](https://github.com/user-attachments/assets/0cc23802-2df5-42e9-a56c-9ebd581dd1b3)
![image](https://github.com/user-attachments/assets/ac10be71-807d-4105-aa79-7cea86ff1034)
![image](https://github.com/user-attachments/assets/f59af344-2cec-4c4e-adc9-ff5e4e125a5e)

---

## 🔍 Features

- 📋 Input 30 diagnostic features from breast cancer cell nuclei
- 🧠 Predict tumor classification: **Malignant** or **Benign**
- 📈 PCA-based 2D visualization with diagnosis clusters
- 🔐 Securely logs data to **Firestore**

---

## 📦 Tech Stack

| Component        | Tool/Library             |
|------------------|--------------------------|
| Frontend UI      | Streamlit                |
| ML Model         | scikit-learn (SVM + PCA) |
| Visualization    | Plotly Express           |
| Data Storage     | Firebase Firestore       |
| Hosting          | Streamlit Cloud          |
