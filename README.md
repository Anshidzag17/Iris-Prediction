# 🌸 Iris Flower Classification Web App

This is a simple **machine learning web app** that predicts the species of an Iris flower based on its **sepal and petal dimensions**. It uses a trained ML model with Flask for the frontend and backend and is now fully **Dockerized for smooth deployment**.

---

## 📊 About the Project

- **Dataset**: Classic Iris dataset from the UCI Machine Learning Repository.
- **Model Algorithm Used**: RandomForest / Logistic Regression / SVM *(choose based on your implementation)*.
- **Training**: Performed using `scikit-learn`.
- **Accuracy**: Achieves approximately **95–98%** accuracy depending on the selected model.

---

## 🚀 Features

- User-friendly web interface built with **Flask**
- Input: Sepal and Petal Length & Width
- Output: Predicted species - `Setosa`, `Versicolor`, or `Virginica`
- Fully containerized using **Docker**

---

## 🧠 ML Workflow Summary

- Data preprocessing and training handled in `prediction.ipynb`
- Model saved as `model.pkl`
- Loaded in `app.py` to make real-time predictions

---

## 🐳 Dockerized Deployment

### 🔨 1. Build the Docker Image

```bash
docker build -t iris-flask-app .
