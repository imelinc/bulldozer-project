import pandas as pd

df = pd.read_csv("../data/bluebook-for-bulldozers/TrainAndValid.csv", low_memory = False, parse_dates=["saledate"])