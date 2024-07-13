

# Credit Card Fraud Detection

This repository contains a project aimed at detecting fraudulent transactions using machine learning techniques. The project leverages Python and various data science libraries to analyze and classify transactions as fraudulent or non-fraudulent.

## Project Overview

Credit card fraud detection is critical for financial institutions to prevent losses and protect customers. This project uses a dataset of transactions to train a machine learning model that can predict the likelihood of fraud.

## Dataset

The dataset used in this project is highly unbalanced, with a much smaller number of fraudulent transactions compared to non-fraudulent ones. It includes features that have been transformed with PCA for confidentiality reasons.

## Key Steps

1. **Data Preprocessing**:
    - Handling missing values
    - Scaling features
    - Splitting the data into training and testing sets

2. **Exploratory Data Analysis (EDA)**:
    - Visualizing the distribution of classes
    - Understanding feature correlations

3. **Modeling**:
    - Training various machine learning models (Logistic Regression, Decision Trees, Random Forest, etc.)
    - Evaluating model performance using metrics such as precision, recall, F1-score, and ROC-AUC

4. **Model Evaluation**:
    - Cross-validation
    - Hyperparameter tuning
    - Analyzing model performance on the test set

## Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/ShaluPriya-R01/CreditCardFraudDetection.git
    cd CreditCardFraudDetection
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```sh
    jupyter notebook Credit Card Fraud Detection.ipynb
    ```

## Results

The project showcases the performance of different models and highlights the importance of using the right evaluation metrics in imbalanced datasets. The final model provides a robust solution for detecting fraudulent transactions.



