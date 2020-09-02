# Import for data analysis

import numpy as np
import pandas as pd


# Visualization import
import matplotlib.pyplot as plt


# Machinelearning import
from sklearn.linear_model import LogisticRegression




# Acquire Data
train_df = pd.read_csv('./Data/train.csv')
test_df = pd.read_csv('./Data/test.csv')

# Analyze data
# Below command helps to determine nulltype data
#train_df.info()

#categorical features

data_distribution_all = train_df.describe(include = 'all')
data_distribution_surv = train_df.describe(percentiles=[.61, .62,0.63,0.64]) 
data_distribution_inc = train_df.describe(include=['O'])       

# Numerical features






# Mixed Features




# Features with Errors


