import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

# Loading the dataset with parsing dates
df= pd.read_csv("../data/bluebook-for-bulldozers/TrainAndValid.csv", low_memory=False, parse_dates=["saledate"])

# the target variable is 'SalePrice'

#Let's add some new features to the dataset
# We can extract few things from the 'saledate' column
df["saleYear"] = df.saledate.dt.year
df["saleMonth"] = df.saledate.dt.month
df["saleDay"] = df.saledate.dt.day
df["saleDayOfWeek"] = df.saledate.dt.dayofweek
df["saleDayOfYear"] = df.saledate.dt.dayofyear

# And we drop the saledate column
df = df.drop("saledate", axis = 1)

# Then, we're going to handle the missing values
for col in df.columns:
    if df[col].isna().sum() > 330158: # 80% of the data is missing
        df = df.drop(col, axis=1)
    else:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())
            

