# OIBSIP
# Iris Flower Classification

## Oasis Infobyte Internship — Data Science

### Project Overview
This project uses Machine Learning to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

### Dataset
- Dataset: Iris dataset from scikit-learn
- Total samples: 150
- Features: 4
- Classes: Setosa, Versicolor, Virginica
- Missing values: None

### Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

### Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

### Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.33% |
| Random Forest | 90.00% |

**Best-performing model:** Logistic Regression, with 93.33% accuracy on the test set.

### Visualizations

#### Iris Scatter Plot
![Iris Scatter Plot](outputs/iris_scatter_plot.png)

#### Logistic Regression Confusion Matrix
![Logistic Regression Confusion Matrix](outputs/logistic_regression_confusion_matrix.png)

#### Random Forest Confusion Matrix
![Random Forest Confusion Matrix](outputs/random_forest_confusion_matrix.png)

### How to Run

1. Install Python.
2. Install the required libraries:

   `python -m pip install pandas matplotlib scikit-learn`

3. Run the script:

   `python iris_classification.py`

### Conclusion
Both models successfully classified Iris flowers. Logistic Regression achieved higher accuracy than Random Forest on this train-test split.

This project demonstrates dataset exploration, data visualization, model training, and evaluation using classification metrics.