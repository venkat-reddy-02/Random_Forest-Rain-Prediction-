# Rain Tomorrow Prediction using Random Forest

## 📌 Project Overview

This project builds a **machine learning classification model** to
predict whether it will rain tomorrow based on historical weather
observations.

The project uses the **Australian weather dataset (`weatherAUS.csv`)**
and a **Random Forest Classifier**. The notebook covers exploratory data
analysis (EDA), missing-value treatment, date feature engineering,
categorical encoding, train-test splitting, model training, and
evaluation.

## 🎯 Objective

The objective is to predict the target variable:

-   **`RainTomorrow`**
    -   `Yes` → Rain is expected tomorrow
    -   `No` → Rain is not expected tomorrow

This is a **binary classification** problem.

## 📊 Dataset

The dataset contains **142,193 records and 24 original columns**.

The weather observations include information such as:

-   Location
-   Minimum and maximum temperature
-   Rainfall
-   Evaporation
-   Sunshine
-   Wind direction and wind speed
-   Humidity
-   Atmospheric pressure
-   Cloud coverage
-   Temperature at 9 AM and 3 PM
-   Rainfall today
-   Rainfall tomorrow

The original dataset contains missing values, particularly in features
such as `Evaporation`, `Sunshine`, `Cloud9am`, `Cloud3pm`, and
pressure-related measurements.

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook

## 🤖 Machine Learning Algorithm

### Random Forest Classifier

The project uses:

``` python
RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    random_state=25
)
```

The model is trained as a binary classifier with the classes:

``` text
No
Yes
```

## 🔄 Project Workflow

The notebook follows these main steps:

``` text
Load Dataset
     ↓
Exploratory Data Analysis
     ↓
Check Data Types
     ↓
Check Missing Values
     ↓
Check Duplicate Records
     ↓
Statistical Analysis
     ↓
Handle Missing Values
     ↓
Date Feature Engineering
     ↓
Remove Unnecessary Columns
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
One-Hot Encoding
     ↓
Random Forest Model
     ↓
Prediction
     ↓
Model Evaluation
```

## 🔍 Exploratory Data Analysis

The notebook performs:

### Dataset Inspection

-   `df.head()`
-   `df.info()`
-   `df.shape`
-   `df.size`

### Missing-Value Analysis

Missing values are identified using:

``` python
df.isnull().sum()
```

and their percentages are calculated using:

``` python
df.isnull().sum() / len(df) * 100
```

### Duplicate Check

Duplicate records are checked using:

``` python
df.duplicated().sum()
```

The notebook found **0 duplicate records**.

### Statistical Analysis

Numerical features are examined using:

``` python
df.describe()
```

Categorical features are examined using:

``` python
df.describe(include=str)
```

## 🧹 Data Preprocessing

### 1. Missing Values

Numerical columns are filled using their **median values**:

``` python
for col in numerical:
    med = df[col].median()
    df[col] = df[col].fillna(med)
```

Categorical missing values are filled with selected mode values:

``` text
RainToday   → No
WindDir9am  → N
WindDir3pm  → SE
WindGustDir → W
```

### 2. Date Feature Engineering

The `Date` column is converted into a datetime format and split into:

-   `Year`
-   `Month`
-   `Day`

``` python
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
```

The original `Date` column is then removed.

### 3. Removing `RISK_MM`

The `RISK_MM` feature is removed before modeling:

``` python
df.drop(['RISK_MM'], inplace=True, axis=1)
```

### 4. Feature and Target Separation

The target variable is:

``` python
y = df['RainTomorrow']
```

The input features are:

``` python
X = df.drop('RainTomorrow', axis=1)
```

### 5. Train-Test Split

The data is divided into training and testing sets using:

``` python
train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=21
)
```

This creates a **70% training set and 30% testing set**.

### 6. One-Hot Encoding

Categorical variables are converted into numerical features using
one-hot encoding.

Encoded features include:

-   `Location`
-   `WindGustDir`
-   `WindDir9am`
-   `WindDir3pm`
-   `RainToday`

After encoding, the Random Forest model uses **118 input features**.

## 🌲 Model Training

A Random Forest Classifier is used to train the model:

``` python
model_S = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    random_state=25
)

model_S.fit(X_train_S, y_train_S)
```

The Random Forest consists of multiple decision trees whose predictions
are combined to perform the final classification.

## 📈 Model Evaluation

The trained model is evaluated using:

``` python
y_pred_S = model_S.predict(X_test)

accuracy_score(y_test, y_pred_S)
```

### Accuracy

The recorded test accuracy is:

**84.20%**

``` text
Accuracy: 0.8419757138168691
```

### Classification Report

  Class            Precision   Recall   F1-Score      Support
  -------------- ----------- -------- ---------- ------------
  No                    0.90     0.90       0.90       33,138
  Yes                   0.64     0.65       0.65        9,520
  **Accuracy**                          **0.84**   **42,658**
  Macro Avg             0.77     0.77       0.77       42,658
  Weighted Avg          0.84     0.84       0.84       42,658

The results show that the model performs differently for the two
classes, with stronger classification metrics for `No` than for `Yes`.

## 📁 Project Structure

A suggested GitHub repository structure is:

``` text
Rain-Tomorrow-Prediction/
│
├── Rain_Prediction(Random_Forest).ipynb
├── README.md
├── weatherAUS.csv
└── requirements.txt
```

> If the dataset is not included in the repository, update the
> notebook's dataset path before running it.

## ▶️ How to Run the Project

### 1. Clone the Repository

``` bash
git clone <your-repository-url>
cd Rain-Tomorrow-Prediction
```

### 2. Install Dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or, if a `requirements.txt` file is provided:

``` bash
pip install -r requirements.txt
```

### 3. Start Jupyter Notebook

``` bash
jupyter notebook
```

### 4. Open the Notebook

Open:

``` text
Rain_Prediction(Random_Forest).ipynb
```

### 5. Update the Dataset Path

The notebook currently loads the dataset using a local Windows path.
Change it to the location of `weatherAUS.csv` on your system, for
example:

``` python
df = pd.read_csv("weatherAUS.csv")
```

### 6. Run All Cells

Execute the notebook from top to bottom to reproduce the preprocessing,
model training, predictions, and evaluation.

## 📌 Model Parameters

  Parameter                                               Value
  ---------------------------------- --------------------------
  Algorithm                            Random Forest Classifier
  Number of Trees (`n_estimators`)                          150
  Maximum Tree Depth (`max_depth`)                           20
  Random State                                               25
  Test Size                                                 30%
  Training Size                                             70%
  Input Features After Encoding                             118
  Number of Classes                                           2
  Test Accuracy                                          84.20%

## 💡 Key Learnings

This project demonstrates practical machine learning concepts including:

-   Exploratory Data Analysis
-   Missing-value handling
-   Numerical and categorical feature identification
-   Median and mode imputation
-   Date feature engineering
-   One-hot encoding
-   Train-test splitting
-   Random Forest classification
-   Model prediction
-   Accuracy evaluation
-   Precision, recall, and F1-score interpretation
-   Feature importance available from the trained Random Forest model

## 🚀 Future Improvements

Possible extensions to the project include:

-   Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
-   Comparing Random Forest with Logistic Regression, Decision Tree,
    XGBoost, or other classifiers
-   Using cross-validation
-   Evaluating ROC-AUC and PR-AUC
-   Investigating class imbalance
-   Experimenting with class weights or resampling techniques
-   Performing feature selection
-   Comparing different preprocessing strategies
-   Building a prediction interface or web application

## 📓 Notebook

The complete implementation and analysis are available in:

``` text
Rain_Prediction(Random_Forest).ipynb
```

## 👤 Author

**Reddy Venkata Rao**

B.Tech -- Electronics and Communication Engineering

------------------------------------------------------------------------

⭐ If you find this project useful, consider giving the repository a
star.
