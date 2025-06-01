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



