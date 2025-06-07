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

# Then, we're going to turn everything into numeric values so the model can understand it

