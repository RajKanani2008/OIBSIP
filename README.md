# Iris Flower Classification 🌸

## Oasis Infobyte Internship — Data Science

### Project Overview

This project uses Machine Learning classification algorithms to predict the species of an Iris flower based on its sepal and petal measurements.

The project classifies Iris flowers into three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Two machine learning models are implemented and evaluated: Logistic Regression and Random Forest Classifier.

## Dataset

The Iris dataset is loaded from `scikit-learn`.

- **Total samples:** 150
- **Input features:** 4
- **Target classes:** 3
- **Missing values:** None

### Features

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as a classification algorithm to predict the species of an Iris flower.

### 2. Random Forest Classifier

Random Forest combines multiple decision trees to make classification predictions.

## Model Evaluation and Results

Both models were evaluated on the same test set.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.33% |
| Random Forest Classifier | 90.00% |

**Best-performing model:** Logistic Regression, with 93.33% test accuracy.

The results reflect this particular train-test split and may vary if the split or model settings change.

## Project Visualizations

### Iris Scatter Plot

This visualization helps explore relationships between Iris flower measurements.

![Iris Scatter Plot](outputs/iris_scatter_plot.png)

### Logistic Regression — Confusion Matrix

This confusion matrix shows the correct and incorrect predictions made by the Logistic Regression model.

![Logistic Regression Confusion Matrix](outputs/logistic_regression_confusion_matrix.png)

### Random Forest — Confusion Matrix

This confusion matrix shows the correct and incorrect predictions made by the Random Forest Classifier.

![Random Forest Confusion Matrix](outputs/random_forest_confusion_matrix.png)

## Project Structure

```text
DataScience-Task1-Iris-Flower-Classification/
├── iris_classification.py
├── README.md
└── outputs/
    ├── iris_scatter_plot.png
    ├── logistic_regression_confusion_matrix.png
    └── random_forest_confusion_matrix.png
```

## Installation and Setup

### Prerequisites

Install Python on your computer before running the project.

### Step 1: Install Required Libraries

Open a terminal in the project folder and run:

```bash
python -m pip install pandas matplotlib scikit-learn
```

### Step 2: Run the Project

```bash
python iris_classification.py
```

The script trains the classification models, evaluates their performance, and generates visualizations.

## Key Learnings

- Loading and exploring a machine learning dataset
- Understanding features and target classes
- Training classification models
- Evaluating model accuracy and classification performance
- Creating scatter plots and confusion matrices
- Comparing the performance of different machine learning algorithms

## Conclusion

Both Logistic Regression and Random Forest successfully classified Iris flower species. Logistic Regression achieved the higher test accuracy of 93.33%, compared with 90.00% for Random Forest.

This project demonstrates a basic machine learning classification workflow, from dataset loading and model training to evaluation and visualization.

---

**Project:** Iris Flower Classification  
**Internship Organization:** Oasis Infobyte  
**Track:** Data Science
