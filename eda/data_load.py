import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_data_prepare_x(path):
    data = pd.read_csv(path)
    
    numeric_columns = data[['SibSp','Parch','Fare','Pclass']]
    dummy_data = pd.get_dummies(data[['Sex','Embarked']])
    
    x = numeric_columns.join(dummy_data)
    return x


def scale_data(x):
    sc = StandardScaler()
    return sc.fit_transform(x)


def plot_histograms(data):
    for var in ['Fare','SibSp', 'Parch']:
        plt.hist(data[var], bins=30)
        plt.title(var+' Histogram')
        plt.show()