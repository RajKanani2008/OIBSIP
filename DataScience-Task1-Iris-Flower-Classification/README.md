@'
# Iris Flower Classification

## Oasis Infobyte Internship — Data Science

### Project Overview
This project uses Machine Learning to classify Iris flowers into three species: Setosa, Versicolor, and Virginica, based on sepal and petal measurements.

### Dataset
- **Dataset:** Iris dataset from scikit-learn
- **Total samples:** 150
- **Features:** 4
- **Classes:** 3
- **Missing values:** None

### Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

### Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

### Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.33% |
| Random Forest Classifier | 90.00% |

**Best-performing model:** Logistic Regression, with 93.33% accuracy on the test set.

### Visualizations

#### Iris Scatter Plot
![Iris Scatter Plot](DataScience-Task1-Iris-Flower-Classification/outputs/iris_scatter_plot.png)

#### Logistic Regression Confusion Matrix
![Logistic Regression Confusion Matrix](DataScience-Task1-Iris-Flower-Classification/outputs/logistic_regression_confusion_matrix.png)

#### Random Forest Confusion Matrix
![Random Forest Confusion Matrix](DataScience-Task1-Iris-Flower-Classification/outputs/random_forest_confusion_matrix.png)

### How to Run

Install the required libraries:

```bash
python -m pip install pandas matplotlib scikit-learn