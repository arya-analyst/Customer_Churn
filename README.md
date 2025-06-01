# Customer Churn Prediction using ML models

## Objective
This project involves a comprehensive analysis and a beginner's guide to creating a ML prediction model by comparing different ML models with each other.

### Getting started with the basics

#### Importing the dependencies:

``` python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

#### Importing the dataset:
``` python
#Dataset Source - Forage x PWC Power BI Job Simulation
df = pd.read_csv(r"C:\Users\ARYA GUPTA\Desktop\Projects Data\02 Churn-Dataset.csv")
```
#### Printing unique values in all the columns

``` python
for col in df.columns:
    print(col, df[col].unique())
    print("*" * 50)
```

#### separating all the columns on which we don't want to apply Label encoder or one hot encoder;
```python
numerical_features_list = ['tenure', 'MonthlyCharges', 'TotalCharges', 'numAdminTickets', 'numTechTickets']
```
#### Converting 'TotalCharges' to float data type

``` python
# checking the no. of entries where TotalCharges is null. We can observe that all the rows having missing TotalCharges have tenure 0.
# which means that this are new customers and hence they don't have any previous charges adding up for them.

len(df[df['TotalCharges'] == ' '])

# replacing the " " empty space with float 0.0 

df['TotalCharges'] = df['TotalCharges'].replace(" ", "0.0")

# converting the object to float data type

df['TotalCharges'] = df['TotalCharges'].astype(float)
```

#### 
``` python
# checking the class distribution of the target class
# we can check that the data isn't balanced here and we need to upsample the data to give the model enough room to learn

df['Churn'].value_counts()
```

#### Dropping unnecessary columns
```python
df.drop(columns = ['numAdminTickets', 'numTechTickets'], inplace = True)
```

###  Exploratory Data Analysis

``` python
def plot_histogram(df, column_name):
    plt.figure(figsize = (5, 5))
    sns.histplot(df[column_name], kde = True)
    plt.title(f"Distribution of {column_name}")

    # calculating the mean and median values for the columns
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()

    # adding vertical lines for mean and median
    plt.axvline(col_mean, color = 'red', linestyle = "--", label = 'Mean')
    plt.axvline(col_median, color = 'green', linestyle = "-", label = 'Median')

    plt.legend()
    
    plt.show()
```
``` python
plot_histogram(df, 'tenure')
```

#### Plotting BoxPlots

``` python
def plot_boxplot(df, column_name):
    plt.figure(figsize = (5, 5))
    sns.boxplot( y = df[column_name])
    plt.title(f"Box Plot for {column_name}")
    plt.ylabel(column_name)
    plt.show()
```
```python
plot_boxplot(df, 'MonthlyCharges')
```

#### Correlation Heatmap

``` python
plt.figure(figsize = (8, 5))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot =  True, cmap = 'coolwarm', fmt=".2f")  #we are specifying a list in which all the column names have been specified
plt.title('Correlation Heatmap')
plt.show()
```

``` python
object_cols = df.select_dtypes(include = 'object').columns.to_list()
object_cols = ['SeniorCitizen'] + object_cols

object_cols
```

#### Plotting CountPlots
```python
for col in object_cols:
    plt.figure(figsize=(6, 3))
    ax = sns.countplot(x=df[col])
    plt.title(f"Count Plot for {col}")

    # Add count numbers on each bar
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2.,  # Horizontal position: center of the bar
            p.get_height() + 1.0,            # Vertical position: slightly above the bar
            int(p.get_height()),             # Text: count number
            ha='center'                      # Horizontal alignment: center
        )

    plt.show()
```

### Data Preprocessing

``` python
df["Churn"] = df["Churn"].replace({"Yes" : 1, "No" : 0})
```

#### Importing Pickle

``` python
import pickle
from sklearn.preprocessing import LabelEncoder
```

``` python
# initialize a dictionary to save the encoders

encoders = {}

# apply label encoding and store the encoders

for col in object_cols:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])
    encoders[col] = label_encoder


# save the encoders to a pickle file
with open ("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
```

``` python
encoders
```

### Training and test data split

``` python
from sklearn.model_selection import train_test_split, cross_val_score
```

``` python
x = df.drop(columns = ['Churn'])
y = df["Churn"]
```

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
```

```python
print(x_train.shape)
print(y_train.shape)
```

```python
print(y_train.value_counts())
```



