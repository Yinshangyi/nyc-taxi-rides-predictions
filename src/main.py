from data_processing import DataProcessing
from model import XGBoostModel
from ml_processing import MLProcessing
import pandas as pd

df = pd.read_csv('data/train.csv', parse_dates=['pickup_datetime'], nrows=50000)

# EDA and Data Processing
data_processing = DataProcessing()
taxi_df = data_processing.train_pipeline(df)

# Data Pre-processing for the model fitting
ml_processing = MLProcessing()
X_train, X_test, y_train, y_test = ml_processing.transform(taxi_df)

# Train the model
model = XGBoostModel()
model.grid_search(X_train, X_test, y_train, y_test)


