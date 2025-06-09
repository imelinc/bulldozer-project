import pandas as pd

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

for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content): # if the column has numeric data types
        if pd.isnull(content).sum():
            df[label + "_is_missing"] = pd.isnull(content)
            df[label] = content.fillna(content.median())
    elif not pd.api.types.is_numeric_dtype(content): # if the column has non-numeric data types
        df[label + "_is_missing"] = pd.isnull(content)
        df[label] = pd.Categorical(content).codes + 1


