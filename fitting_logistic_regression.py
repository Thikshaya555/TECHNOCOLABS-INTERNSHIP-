import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

df = pd.read_csv('cleaned_data.csv')
features_response = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default payment next month']
from sklearn.model_selection import train_test_split
X = df[['PAY_1', 'LIMIT_BAL']]
y = df['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver = 'liblinear')

lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
import pickle
pickle.dump(lr_model,open('lr_model.pkl','wb'))

