import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from extra_functions.funcs import evaluate_model
from sklearn.model_selection import RandomizedSearchCV
import joblib

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
print("\nBaseline Model Evaluation Scores:")
for score_name, score_value in scores.items():
    print(f"{score_name}: {score_value:.2f}")
print("\n")

# Let's try to improve our model using RandomizedSearchCV
# Define the hyperparameter grid
grid = {
    "n_estimators": np.arange(10,100, 10),
    "max_depth": [None, 3,5,10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
    "max_features": [0.5, 1, "sqrt", "log2"],
    "max_samples": [10000]
}

# Let's build the RandomizedSearchCV model
rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, random_state=42),
                                                   param_distributions=grid,
                                                   n_iter=100,
                                                   cv=5,
                                                   verbose=True,
                                                   error_score= "raise")

# Fit the model
rs_model.fit(X_train, y_train)
# Evaluate the model
rs_scores = evaluate_model(rs_model, X_train, y_train, X_valid, y_valid)
# Print the scores
print("\nRandomized Search Model Evaluation Scores:")
for score_name, score_value in rs_scores.items():
    print(f"{score_name}: {score_value:.2f}")
print("\n")

# Well, the tuned model is worse than the basline model, so let's stick with the baseline model
# Save the model
joblib.dump(model, "//models/bulldozer_model.pkl")

# now lets take a look at the feature importances
df_features = pd.DataFrame({"features": X_train.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots()
ax.barh(df_features["features"].head(20), df_features["importance"].head(20), color="blue")
ax.set_ylabel("Features")
ax.set_xlabel("Feature Importance")
ax.invert_yaxis()
ax.set_titple("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("//imgs/bulldozer_feature_importances.png")

