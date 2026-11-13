# 🌸 Iris Flower Classification Using Machine Learning

## 📌 Project Overview

This project is part of **Task 1 of the CodeAlpha Data Science Internship**.

The Iris Flower Classification project is a Machine Learning-based classification system that predicts the species of an iris flower based on its physical measurements.

The project uses the famous Iris dataset, which contains measurements of three different species of iris flowers. A Support Vector Machine (SVM) algorithm is implemented to classify flowers into their respective species.

The main objective of this project is to understand and implement the complete Machine Learning workflow, including data exploration, preprocessing, feature scaling, model training, testing, and evaluation.

---

## 🎯 Project Objectives

The main objectives of this project are:

- To understand and explore the Iris dataset.
- To perform data cleaning and preprocessing.
- To analyze the features and target variable.
- To understand and handle duplicate feature measurements appropriately.
- To split the dataset into training and testing sets.
- To apply feature scaling using StandardScaler.
- To build a Machine Learning classification model.
- To train the model using Support Vector Machine (SVM).
- To evaluate model performance using accuracy.
- To predict iris flower species based on their measurements.

---

## 📂 Project Structure

The project contains the following files:

```text
CodeAlpha_Iris_Flower_Classification/
│
├── Iris.csv
│
├── iris_classification.py
│
└── README.md
```

### File Description

| File Name | Description |
|---|---|
| Iris.csv | Dataset containing iris flower measurements and species labels. |
| iris_classification.py | Python source code for data preprocessing, model training, testing, and evaluation. |
| README.md | Documentation explaining the project, methodology, and results. |

---

## 📊 Dataset Description

The project uses the **Iris Flower Dataset**, one of the most popular datasets used in Machine Learning and Data Science.

The dataset contains 150 flower samples belonging to three different iris species.

### Iris Species

1. Iris Setosa
2. Iris Versicolor
3. Iris Virginica

Each species contains 50 samples.

### Dataset Features

The dataset contains the following columns:

| Feature | Description |
|---|---|
| Id | Unique identification number for each sample. |
| SepalLengthCm | Length of the sepal in centimeters. |
| SepalWidthCm | Width of the sepal in centimeters. |
| PetalLengthCm | Length of the petal in centimeters. |
| PetalWidthCm | Width of the petal in centimeters. |
| Species | Target variable representing the flower species. |

### Dataset Information

- Total records: 150
- Total species: 3
- Samples per species: 50
- Input features: 4
- Target variable: Species
- Missing values: None

The `Id` column is removed before model training because it is only an identifier and does not provide meaningful information for predicting flower species.

---

## 🧹 Data Cleaning and Duplicate Rows Handling

During data preprocessing, duplicate feature measurements were carefully examined.

Some flower samples contain identical measurements across the input features. However, identical measurements do not necessarily mean that the records are duplicate or invalid.

Different flower samples can naturally have the same measurements.

Therefore, automatically removing these records could result in the loss of valid observations.

### Decision Taken

- All 150 original records were retained.
- No records were removed based only on identical feature measurements.
- The `drop_duplicates()` function was not applied.
- The dataset was checked for missing values.
- The unnecessary `Id` column was removed before model training.

**Reason:** The original observations were preserved to avoid losing potentially valid flower samples without sufficient evidence that they were erroneous.

This approach ensures that the dataset remains intact and maintains the original sample distribution.

---

## 🛠️ Technologies and Libraries Used

The project is developed using Python and the following libraries:

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning model development |
| StandardScaler | Feature standardization |
| SVC | Support Vector Machine classification |

---

## 🤖 Machine Learning Algorithm

### Support Vector Machine (SVM)

Support Vector Machine is a supervised Machine Learning algorithm commonly used for classification and regression problems.

In this project, an SVM classifier with a **Linear Kernel** is used to classify iris flowers into three different species.

The algorithm learns patterns from the flower measurements and identifies the decision boundaries that separate the different species.

### Why SVM?

- Effective for classification problems.
- Works well with numerical datasets.
- Can handle multiple classes.
- Performs well on small and structured datasets.
- Can create effective decision boundaries between different classes.

---

## ⚙️ Project Workflow

The project follows these steps:

### 1. Importing Required Libraries

The necessary Python libraries are imported for data manipulation, preprocessing, model development, and evaluation.

### 2. Loading the Dataset

The Iris dataset is loaded using the Pandas library.

```python
import pandas as pd

df = pd.read_csv("Iris.csv")
```

### 3. Data Exploration

The dataset is explored to understand its structure and characteristics.

The following operations are performed:

- Displaying the first few records.
- Checking the dataset dimensions.
- Examining column names and data types.
- Checking for missing values.
- Understanding the distribution of flower species.

### 4. Data Preprocessing

Data preprocessing is performed before training the model.

The following steps are included:

- Removing the unnecessary `Id` column.
- Separating input features and target labels.
- Checking for missing values.
- Retaining all 150 original records.
- Avoiding automatic removal of identical feature measurements.

The four numerical measurements are used as input features, while `Species` is used as the target variable.

### 5. Splitting the Dataset

The dataset is divided into training and testing sets using an 80:20 ratio.

- Training data: 80% (120 samples).
- Testing data: 20% (30 samples).

A stratified train-test split is used to maintain the proportion of all three flower species in both datasets.

### 6. Feature Scaling

StandardScaler is used to standardize the numerical input features.

Feature scaling helps ensure that features with different numerical ranges contribute appropriately to the model.

A Scikit-learn Pipeline is used to combine feature scaling and model training.

This also ensures that the scaler is fitted using training data rather than the complete dataset, helping prevent data leakage.

### 7. Model Training

A Support Vector Machine classifier with a linear kernel is trained using the training dataset.

The model learns patterns from the flower measurements to classify the iris species.

### 8. Model Evaluation

The trained model is evaluated using the testing dataset.

Accuracy is calculated to measure the proportion of correctly classified flower samples.

---

## 📈 Model Performance and Results

The Support Vector Machine classifier achieved the following result in the current experiment:

| Evaluation Metric | Result |
|---|---|
| Algorithm | Support Vector Machine (SVM) |
| Kernel | Linear |
| Total Dataset Samples | 150 |
| Training Samples | 120 |
| Testing Samples | 30 |
| Accuracy | 100% |

### Accuracy Result

**Model Accuracy: 100%**

The model correctly classified all 30 samples in the selected test dataset.

This result represents the model's performance on this particular train-test split. It does not guarantee 100% accuracy on every future dataset or different train-test split.

---

## 💻 Installation and Execution

Follow these steps to run the project on your local computer.

### Prerequisites

Make sure you have installed:

- Python 3.x
- Visual Studio Code
- Required Python libraries

### Step 1: Download the Repository

Download or clone the repository to your local computer.

### Step 2: Open the Project Folder

Open the project folder in Visual Studio Code.

### Step 3: Install Required Libraries

Open the VS Code terminal and execute:

```bash
pip install pandas numpy scikit-learn
```

### Step 4: Run the Python File

Execute the following command:

```bash
python iris_classification.py
```

The program will:

- Load the Iris dataset.
- Perform data preprocessing.
- Split the dataset into training and testing sets.
- Apply feature scaling.
- Train the SVM classification model.
- Evaluate model performance.
- Display the classification results.

---

## 📚 Key Learnings

Through this project, I gained practical experience in:

- Understanding structured datasets.
- Performing data exploration and preprocessing.
- Handling unnecessary dataset columns.
- Understanding duplicate feature measurements.
- Learning why valid observations should not be removed automatically.
- Splitting datasets into training and testing sets.
- Applying stratified sampling.
- Performing feature scaling using StandardScaler.
- Building Machine Learning Pipelines.
- Implementing Support Vector Machine classification.
- Evaluating model performance using accuracy.
- Working with Python and Scikit-learn.
- Understanding the complete Machine Learning workflow.

---

## 🚀 Future Improvements

The project can be further improved by:

- Experimenting with different classification algorithms such as Decision Tree, Random Forest, and K-Nearest Neighbors.
- Comparing linear and nonlinear SVM kernels.
- Performing hyperparameter tuning.
- Evaluating the model using precision, recall, and F1-score.
- Applying cross-validation for more reliable performance evaluation.
- Developing a simple web application for interactive iris flower classification.
- Improving model evaluation using additional performance metrics.

---

## 👨‍💻 Author

**Data Science Intern**

**Internship Organization:** CodeAlpha

**Project:** Iris Flower Classification

**Task:** Task 1 – Data Science Internship

---

## 🏁 Conclusion

The Iris Flower Classification project demonstrates how Machine Learning can be applied to solve a multi-class classification problem.

By using data exploration, preprocessing, feature scaling, and a Support Vector Machine classifier, the project successfully classifies iris flowers into three different species.

The decision to retain all 150 original records ensures that potentially valid observations are not unnecessarily removed during preprocessing.

The project provided valuable hands-on experience in Python programming, data preprocessing, supervised Machine Learning, and model evaluation.

**This project was completed as part of the CodeAlpha Data Science Internship – Task 1.**
