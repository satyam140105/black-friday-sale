# Black Friday Sales Prediction

## Project Overview
This project aims to predict the purchase amount of customers during Black Friday sales using historical retail data. The goal is to help retail businesses optimize their pricing strategies to maximize profits.

The dataset contains customer demographics, product details, and purchase amounts. Various machine learning models were implemented to predict the purchase amount, and the best-performing model was selected based on evaluation metrics.

---

## Table of Contents
1. [Project Introduction](#project-introduction)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Conclusion](#conclusion)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Introduction
Black Friday is one of the biggest shopping events of the year, and retailers aim to maximize their profits by optimizing product pricing. This project uses machine learning to predict customer purchase amounts based on historical sales data. The predictions can help retailers set personalized offers and pricing strategies.

---

## Dataset Description
The dataset contains the following features:

| Column Name                  | Description                                      |
|------------------------------|--------------------------------------------------|
| `User_ID`                    | Unique ID of the customer                        |
| `Product_ID`                 | Unique ID of the product                         |
| `Gender`                     | Gender of the customer (M/F)                     |
| `Age`                        | Age group of the customer                        |
| `Occupation`                 | Occupation of the customer (masked)              |
| `City_Category`              | Category of the city (A, B, C)                   |
| `Stay_In_Current_City_Years` | Number of years in the current city              |
| `Marital_Status`             | Marital status of the customer (0 = Single, 1 = Married) |
| `Product_Category_1`         | Main product category (masked)                   |
| `Product_Category_2`         | Secondary product category (masked)              |
| `Product_Category_3`         | Tertiary product category (masked)               |
| `Purchase`                   | Purchase amount (Target Variable)                |

---

## Exploratory Data Analysis (EDA)
Key insights from the dataset:
- **Gender Distribution**: 75% of purchases are made by male customers.
- **Age Group**: Customers aged 25-40 spend the most.
- **Marital Status**: Single customers spend more than married customers.
- **City Category**: City B contributes the most to overall sales, but City C has the highest average purchase amount.
- **Stay in Current City**: Customers who have lived in the city for 1 year spend the most.

Visualizations are available in the Jupyter Notebook or Python script.

---

## Data Preprocessing
1. **Handling Missing Values**: Filled missing values in `Product_Category_2` and `Product_Category_3` with 0.
2. **Encoding Categorical Variables**: Used `LabelEncoder` for `Gender`, `Age`, and `City_Category`.
3. **Dummy Variables**: Created dummy variables for `Stay_In_Current_City_Years`.
4. **Dropping Irrelevant Columns**: Removed `User_ID` and `Product_ID` as they are not useful for modeling.

---

## Modeling
The following machine learning models were implemented:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **XGBoost Regressor**

The **XGBoost Regressor** performed the best with the lowest RMSE score.

---

## Evaluation Metrics
The models were evaluated using **Root Mean Squared Error (RMSE)**. The results are as follows:

| Model                  | RMSE     |
|------------------------|----------|
| Linear Regression       | 4619.83  |
| Decision Tree Regressor | 3396.35  |
| Random Forest Regressor | 3115.80  |
| XGBoost Regressor       | 2879.33  |

---

## Conclusion
The **XGBoost Regressor** was the best-performing model with an RMSE of **2879.33**. This model can be used to predict customer purchase amounts and help retailers optimize their pricing strategies for Black Friday sales.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/satyam140105/black-friday-sale.git
