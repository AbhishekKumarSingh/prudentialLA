import pandas as pd
import numpy as np
import os, glob
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import csv

def read_data(path, path_test):
    print("reading training data")
    df = pd.read_csv(path)
    le = preprocessing.LabelEncoder()
    le.fit(df['Product_Info_2'])
    df['Product_Info_2'] = le.transform(df['Product_Info_2'])
    df = df.fillna(value=0)
    #print df.dtypes
    print("reading test data")
    df_test = pd.read_csv(path_test)
    le.fit(df_test['Product_Info_2'])
    df_test = df_test.fillna(value=0)
    df_test['Product_Info_2'] = le.transform(df_test['Product_Info_2'])
    return df.values, df_test.values


def train_gbm(path, path_test):
    print("Starting Data Science")
    clf = GradientBoostingClassifier()
    data_train, data_test = read_data(path, path_test)
    X,y = data_train[1:,:-1], data_train[1:,-1:]
    #X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.33)
    #print X_train.shape, X_val.shape, y_train.shape, y_val.shape
    print X.shape, y.shape, data_test.shape
    clf.fit(X, y.ravel())
    print("Data Science done!!")
    y_pred = clf.predict(data_test[1:,:])
    print("Printing shape of prediction")
    print y_pred.shape
    y_pred1 = [int(predict) for predict in y_pred]
    id = [int(test_p) for test_p in data_test[:,0]]
    print("Writing to file")
    with open('prediction.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows([['Id','Response']])
        writer.writerows(zip(id,y_pred1)) 
    print("Ban gaya Data Scientist!!!")

def main():
    train_gbm('./data/train.csv', './data/test.csv')

if __name__=="__main__":
    main()
