import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


## function to load the data
def load_data():
    file_path = os.path.join('../data', 'sample_diabetes_mellitus_data.csv')
    df = pd.read_csv(file_path, index_col=0)
    
    return df


## function remove NA rows using subset
def remove_na(df):
    df = df.dropna(subset=["age", "gender", "ethnicity"])
    
    return df


## function to fill NA with mean
def na_impute_mean(df):
    df["height"] = df["height"].fillna(df["height"].mean())
    df["weight"] = df["weight"].fillna(df["weight"].mean())
    
    return df


## function for ethnicity dummy columns
def encode_ethnicity(df):
    df["ethnicity"] = df["ethnicity"].astype(str)
    df = pd.get_dummies(df, columns=["ethnicity"], prefix="ethnicity")
    
    return df


## function to make gender binary
def binary_gender(df):
    df["gender"] = df["gender"].map({"F":0, "M":1})
    
    return df


## function to split data into train and test
def split_train_test(df, target_column="diabetes_mellitus"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=0.2, train_size=0.8, shuffle=True)


## function for train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model


## function to predict test
def predict(model, X_test):
    pred = model.predict(X_test)

    return pred


## function for accuracy score
def accuracy_sc(y_test, y_pred):
    score = accuracy_score(y_test, y_pred)
    
    return score


## function to predict train
def predict_probababilities(model, X):
    p_proba = model.predict_proba(X)[:, 1]
    
    return p_proba


## function to compute roc_auc
def calc_roc_auc(model, X, y):
    y_predprobabilities = predict_probababilities(model, X)  
    roc_score = roc_auc_score(y, y_predprobabilities)
    
    return roc_score