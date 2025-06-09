import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from extra_functions.funcs import evaluate_model


df = pd.read_csv("../data/bluebook-for-bulldozers/TrainAndValid.csv", low_memory = False, parse_dates=["saledate"])

# We have plenty of missing value sin this dataset, we're goint to handle those in a moment
# First, let's add some features that could be useful for our model

df['saleYear'] = df.saledate.dt.year
df['saleMonth'] = df.saledate.dt.month
df['saleDay'] = df.saledate.dt.day
df['saleDayOfWeek'] = df.saledate.dt.dayofweek
df['saleDayOfYear'] = df.saledate.dt.dayofyear

# Now we can drop the saledate column
df = df.drop("saledate", axis=1)

# Now we're ready to turn everything into numbers, and that is goint to be done by using the pandas "categorical" type

for label, content in df.items():
    if pd.api.types.is_object_dtype(content): # if the column is a 'object' data type
        df[label] = content.astype("category").cat.as_ordered() # convert to categorical type

# Let's handle the missing values now
for col in df.columns:
    if df[col].isna().sum()*100/ len(df) > 70: # if more than 70% of the column is missing, we drop it
        df = df.drop(col, axis = 1)

for label, content in df.items(): # iterate through each column
    if pd.api.types.is_numeric_dtype(content): # if the column has numeric data types
        if pd.isnull(content).sum(): # if there are missing values in the column
            df[label + "_is_missing"] = pd.isnull(content) # create a new column indicating if the value is missing
            df[label] = content.fillna(content.median()) # fill missing values with the median
    elif not pd.api.types.is_numeric_dtype(content): # if the column has non-numeric data types
        df[label + "_is_missing"] = pd.isnull(content) # create a new column indicating if the value is missing
        df[label] = pd.Categorical(content).codes + 1 # convert to categorical codes and add 1 to avoid -1 for missing values

# Ready to build a model
# First we need to split the data into training and validation
df_val = df[df.saleYear == 2012] # validation set (2012 because it is the latest year in the dataset)
df_train = df[df.saleYear != 2012] # training set 

X_train, y_train = df_train.drop(["SalePrice", "SalesID"], axis = 1), df_train.SalePrice
X_valid, y_valid = df_val.drop(["SalePrice", "SalesID"], axis = 1), df_val.SalePrice

# Let's build a model
model = RandomForestRegressor(n_jobs=-1, random_state=42, max_samples=10000)
model.fit(X_train, y_train)

# Let's use the function in "funcs.py" to evaluate our model
scores = evaluate_model(model, X_train, y_train, X_valid, y_valid)

# let's print the scores
for score_name, score_value in scores.items():
    print(f"{score_name}: {score_value:.2f}")

