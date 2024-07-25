# Intrusion-Detection-System
## Problem Statement
Network Intrusion Detection Systems (NIDS) are crucial for monitoring and securing network traffic against unauthorized access and cyber threats. This project aims to develop an advanced NIDS using machine learning techniques to accurately detect and classify malicious activities in real-time, leveraging the UNSW-NB15 dataset to train and evaluate models such as Random Forest, XGBoost, and Decision Trees. The goal is to enhance network security by identifying potential intrusions promptly, minimizing false positives, and ensuring robust protection against evolving cyber threats.

## Project Overview
This project aims to develop an advanced Intrusion Detection System (IDS) using machine learning techniques to analyze network traffic and identify potential threats. The following steps were undertaken to achieve this goal:

### 1. Exploratory Data Analysis (EDA):
  i) Conducted thorough EDA to understand the distribution and relationships within the UNSW-NB15 dataset.                                     
  ii) Visualized various features to identify patterns and anomalies.

### 2. Feature Engineering:
  i) Performed data preprocessing and transformation to create meaningful features.                                                         
  ii) Applied techniques such as target smoothing and encoding for categorical variables.

### 3. Model Building and Tuning:
  i) Built and trained models using algorithms like Random Forest, XGBoost, and Decision Trees.                                                         
  ii) Used Random Search and Grid Search CV for hyperparameter tuning to optimize model performance.

### 4. Pipeline Creation:
  i) Developed a custom scikit-learn pipeline to streamline the prediction process for test data.                                                            
  ii) Ensured efficient data processing and model application in a seamless workflow.

### 5. Deployment:
  i) Deployed the trained IDS model using Flask to create a web application.                                                                    
  ii) Implemented endpoints to handle user inputs and return predictions, redirecting to appropriate pages based on the results.
