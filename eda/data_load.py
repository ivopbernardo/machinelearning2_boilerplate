import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data_prepare_x(path):
    data = pd.read_csv(path)
    
    numeric_columns = data[['SibSp','Parch','Fare','Pclass']]
    dummy_data = pd.get_dummies(data[['Sex','Embarked']])
    
    x = numeric_columns.join(dummy_data)
    return x