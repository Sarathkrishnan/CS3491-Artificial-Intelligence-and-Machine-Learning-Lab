

### 1️⃣ **Load Dataset**
```python
import pandas as pd

df = pd.read_csv("loan_data.csv")  # Load loan data into a DataFrame
df.head()  # Show the first 5 rows
df.info()  # Display dataset info (columns, data types, missing values)
```
- **Pandas (`pd`)** is used to load a CSV file containing loan data.
- `.head()` previews the dataset.
- `.info()` provides a summary of the dataset.

---

### 2️⃣ **Visualize Loan Purpose vs Loan Status**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x="purpose", hue="not.fully.paid")
plt.xticks(rotation=45, ha="right");
```
- **Seaborn (`sns`)** and **Matplotlib (`plt`)** are used for visualization.
- `sns.countplot()` creates a bar plot of **loan purposes**.
- The `hue="not.fully.paid"` differentiates loans that were **fully paid** vs **not fully paid**.
- `plt.xticks(rotation=45)` rotates x-axis labels for readability.

---

### 3️⃣ **Convert Categorical Data into Numerical Data**
```python
pre_df = pd.get_dummies(df, columns=["purpose"], drop_first=True)
pre_df.head()
```
- `pd.get_dummies()` converts **categorical variable** `"purpose"` into multiple binary (0/1) columns.
- `drop_first=True` avoids **dummy variable trap** (removes one category to prevent redundancy).

---

### 4️⃣ **Split Data into Features & Target**
```python
from sklearn.model_selection import train_test_split

X = pre_df.drop("not.fully.paid", axis=1)  # Features
y = pre_df["not.fully.paid"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)
```
- `X` contains **independent variables** (loan features).
- `y` contains the **target variable** (`not.fully.paid` → 1 if not paid, 0 if paid).
- `train_test_split()` splits data into:
  - **67% training data** (`X_train`, `y_train`)
  - **33% test data** (`X_test`, `y_test`)
- `random_state=125` ensures reproducibility.

---

### 5️⃣ **Train a Naïve Bayes Model**
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train);
```
- **Gaussian Naïve Bayes (`GaussianNB`)** is used for classification.
- `.fit(X_train, y_train)` trains the model.

---

### 6️⃣ **Evaluate Model Performance**
```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

y_pred = model.predict(X_test)

accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)
```
- `accuracy_score()` calculates **accuracy**.
- `f1_score()` calculates **F1 score** (better for imbalanced data).
- `model.predict(X_test)` predicts **loan repayment status** for test data.

---

### 7️⃣ **Confusion Matrix**
```python
labels = ["Fully Paid", "Not fully Paid"]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();
```
- **Confusion matrix** helps evaluate model performance:
  - True Positives (TP): Correctly predicted **Not Fully Paid**
  - True Negatives (TN): Correctly predicted **Fully Paid**
  - False Positives (FP): Incorrectly predicted **Not Fully Paid**
  - False Negatives (FN): Incorrectly predicted **Fully Paid**
- `ConfusionMatrixDisplay()` displays the confusion matrix.

---

### **Summary**
✅ Load & preprocess loan dataset  
✅ Visualize loan purpose vs repayment status  
✅ Encode categorical data  
✅ Split into training & test sets  
✅ Train a **Naïve Bayes** model  
✅ Predict and evaluate performance using **accuracy, F1-score, and confusion matrix**  