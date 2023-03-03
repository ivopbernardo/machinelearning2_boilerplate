from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def load_target(path):
    target_var = pd.read_csv(path)['Survived']
    return target_var


def train_model(model, scaled_x, target):

    if model == 'Logistic Regression':
        model_fit = LogisticRegression()
        model_fit.fit(scaled_x, target)
    elif model == 'Random Forest':
        model_fit = RandomForestClassifier()
        model_fit.fit(scaled_x, target)
    else:
        return 'Please specify model, first'
    return model_fit