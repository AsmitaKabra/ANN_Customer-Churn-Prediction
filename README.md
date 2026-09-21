# 🔮 ANN Customer Churn Prediction

An end-to-end **Artificial Neural Network (ANN)** project that predicts whether a customer is likely to churn based on their demographic, account, and service-related information.

The project includes **data preprocessing, exploratory data analysis, ANN model development, evaluation, and an interactive Streamlit web application** for real-time predictions.

## 🚀 Live Demo

🔗 **[Try the Customer Churn Prediction App](https://anncustomer-churn-prediction.streamlit.app/)**

---

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help organizations take preventive actions and improve customer retention.

This project uses an **Artificial Neural Network (ANN)** to learn patterns from historical customer data and predict whether a customer is likely to churn.

### 🎯 Objective

* Predict customer churn using Deep Learning.
* Build and evaluate an ANN classification model.
* Perform appropriate data preprocessing and feature transformation.
* Deploy the trained model as an interactive Streamlit application.

---

## 🧠 Machine Learning Approach

The project follows an end-to-end machine learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
ANN Model
   ↓
Model Evaluation
   ↓
Streamlit Deployment
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

### Tools

* Jupyter Notebook
* Git
* GitHub
* Streamlit Community Cloud

---

## 🏗️ ANN Architecture

The model is built using **TensorFlow/Keras**.

A typical architecture consists of:

```text
Input Features
      ↓
Dense Layer
      ↓
Activation Function
      ↓
Dense Layer
      ↓
Activation Function
      ↓
Output Layer
      ↓
Churn Prediction
```

The network is trained using the processed customer dataset to learn the relationship between customer attributes and churn behavior.

---

## 📊 Features

The model uses customer-related attributes such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Contract
* Payment Method
* Monthly Charges
* Total Charges
* And other relevant customer attributes

---

## 📈 Model Evaluation

The ANN model is evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics help measure how effectively the model identifies customers who are likely to churn.

---

## 🌐 Streamlit Application

The trained ANN model is integrated into a **Streamlit web application**.

Users can enter customer information through an interactive interface and receive a churn prediction.

### Application Flow

```text
User Input
    ↓
Preprocessing
    ↓
Trained ANN Model
    ↓
Prediction Probability
    ↓
Churn / No Churn
```

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AsmitaKabra/ANN_Customer-Churn-Prediction.git
```

### 2. Navigate to the project

```bash
cd ANN_Customer-Churn-Prediction
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📂 Project Structure

```text
ANN_Customer-Churn-Prediction/
│
├── app.py
│
├── churn_model.keras
│
├── Churn_Modelling.csv
│
├── experiments.ipynb
│
├── prediction.ipynb
│
├── label_encoder_gender.pkl
│
├── onehot_encoder_geo.pkl
│
├── scaler.pkl
│
├── requirements.txt
│
└── README.md
```

> The exact folder structure may vary depending on the files included in the repository.

---

## 🔍 Key Learning Outcomes

Through this project, I learned:

* Fundamentals of Artificial Neural Networks.
* Building classification models using TensorFlow/Keras.
* Data preprocessing for Deep Learning.
* Feature scaling and encoding.
* Model evaluation using classification metrics.
* Understanding training and validation performance.
* Integrating a trained Deep Learning model with Streamlit.
* Deploying a Deep Learning application using Streamlit Community Cloud.

---

## 🚀 Future Improvements

Possible improvements include:

* Hyperparameter tuning.
* Experimenting with different ANN architectures.
* Handling class imbalance using appropriate techniques.
* Adding ROC-AUC and Precision-Recall curves.
* Improving prediction probability visualization.
* Adding model explainability using SHAP.
* Containerizing the application using Docker.
* Adding experiment tracking with MLflow.

---

## 👩‍💻 Author

### Asmita Kabra

🎓 B.Tech — Artificial Intelligence & Data Science

🔗 **GitHub:** [AsmitaKabra](https://github.com/AsmitaKabra)

🔗 **LinkedIn:** [Connect with me on LinkedIn](https://www.linkedin.com/in/asmita-kabra/)

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub!
