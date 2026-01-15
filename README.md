# Titanic Survival Prediction using Decision Tree & Discretization


## 📌 Project Overview
This project demonstrates a complete machine learning workflow using the **Titanic dataset**, focusing on how **feature discretization** impacts model performance. A **Decision Tree Classifier** is trained on both raw and discretized features (`Age` and `Fare`) to compare accuracy and behavior.

The project highlights the practical use of:
- Feature engineering
- Quantile-based discretization
- Model evaluation
- Visualization of data transformation

---

## 🎯 Objectives
- Build a Decision Tree classifier for survival prediction
- Apply `KBinsDiscretizer` for feature binning
- Compare model accuracy **before and after discretization**
- Visualize feature distributions
- Perform cross-validation for robust evaluation

---

## 📂 Dataset
- **Source:** Titanic dataset (`train.csv`)
- **Features used:**
  - `Age`
  - `Fare`
- **Target:**
  - `Survived`
- Missing values are removed before training.

---

## 🛠️ Technologies & Libraries
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 🔁 Workflow
1. Load and clean dataset
2. Split data into training and testing sets
3. Train Decision Tree on raw features
4. Discretize features using quantile strategy
5. Retrain model on transformed data
6. Compare accuracy scores
7. Visualize feature distributions
8. Perform 10-fold cross-validation

---

## 🔬 Feature Discretization
- Technique: **KBinsDiscretizer**
- Strategy: `quantile`
- Encoding: `ordinal`
- Number of bins: 15
- Applied separately to `Age` and `Fare` using `ColumnTransformer`

---

## 📊 Model Evaluation
| Model Type | Accuracy |
|-----------|----------|
| Raw Features | ~62% |
| Discretized Features | ~63% |

✔️ Discretization slightly improves model performance.

---

## 📈 Visualizations
- Histograms comparing:
  - Original vs discretized `Age`
  - Original vs discretized `Fare`

These plots help understand how binning changes data distribution.

---

## 🔁 Cross-Validation
- Technique: 10-fold Cross Validation
- Metric: Accuracy
- Ensures model stability and generalization

---

## 🚀 How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn

---

## Key Learnings
Feature engineering can significantly affect model behavior
Discretization is useful for tree-based models
Visualization aids better understanding of transformations
Cross-validation provides reliable performance estimates

