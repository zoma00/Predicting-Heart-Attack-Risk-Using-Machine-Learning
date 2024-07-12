# Predicting-Heart-Attack-Risk-Using-Machine-Learning
This repository explores the use of machine learning to predict heart attack risk factors. It analyzes a dataset of heart failure patients, examining various features like age, medical history, lab results, and the occurrence of death events.

## Heart Attack Failure Prediction - Jupyter Notebook Documentation

This repository contains a Jupyter Notebook for analyzing a dataset on heart attack prediction. The notebook explores the data, performs feature engineering, trains a machine learning model, and evaluates it's  performance.

**Data Description:**

The dataset,  "Heart Failure prediction.csv", contains 299 rows and 13 columns related to heart failure patients. The columns include:

* Demographic information: age, sex
* Medical history: anaemia, diabetes, high_blood_pressure, smoking
* Lab results: creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium
* Target variable: DEATH_EVENT (1: death event, 0: no death event)

**Notebook Content:**

1. **Imports:**
    - Standard libraries: pandas, numpy, matplotlib, seaborn
    - Machine learning libraries: sci-kit-learn for data preprocessing, model training, and evaluation

2. **Data Loading:**
    - Loads the CSV data using pandas.read_csv()

3. **Exploratory Data Analysis (EDA):**
    - Provides basic summary statistics using describe().T
    - Analyzes data distribution and missing values
    - Creates visualizations (count plots, heatmaps) to understand relationships between features and the target variable (DEATH_EVENT)

4. **Data Preprocessing:**
    - Identifies categorical and numerical features
    - Encodes categorical features (if necessary)
    - Handles missing values (if any)
    - Splits the data into training and testing sets using train_test_split()

5. **Model Training:**
    - Uses Logistic Regression for heart attack prediction
    - Applies StandardScaler() for feature scaling

6. **Model Evaluation:**
    - Trains the model on the training set using fit()
    - Makes predictions on the testing set using predict()
    - Evaluates model performance using:
        - Classification report (precision, recall, F1-score)
        - Accuracy score
    - Compares training and testing set performance

**Getting Started:**

1. Clone this repository.
2. Install required libraries using `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open the Jupyter Notebook (e.g., `Heart_Attack_Failure_Prediction.ipynb`)
4. Run the code cells sequentially.

**Note:**

This documentation provides a high-level overview. Refer to the Jupyter Notebook for detailed code explanations and visualizations.
