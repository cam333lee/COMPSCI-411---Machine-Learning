##Assignment 3
##Cameron Lee
##4/29/23
##OpenML Datasets ID's: 44957, 23
##Links: https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise#

from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.metrics import RootMeanSquaredError
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

def airfoil():
    airfoil = datasets.fetch_openml(data_id=44957)
    print(airfoil.feature_names)
    print(airfoil.target)
    
    #Most appropriate number of hidden neurons is sqrt(input layer nodes * output layer nodes)
    #SMALL NUMBER OF NODES (2)
    #kfold cross-validation
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_rmse=[] #list to store mse of each fold
    for train, test in kfolds.split(airfoil.data, airfoil.target):
        #build neural network
        nn1 = models.Sequential()
        nn1.add(layers.Dense(2, activation='relu', input_dim = 5))
        nn1.add(layers.Dense(1))
        nn1.compile(optimizer='rmsprop', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

        #training
        n_epochs = 100
        nn1.fit(airfoil.data.iloc[train], airfoil.target.iloc[train],epochs=n_epochs)

        #testing
        s=nn1.evaluate(airfoil.data.iloc[test], airfoil.data.iloc[test])
        test_fold_rmse.append(s[1])
        print("Fold", len(test_fold_rmse),"RMSE =", s[1])

    print("\nTesting RMSE for all folds:", test_fold_rmse)
    print("\nAverage RMSE for all folds:", np.mean(test_fold_rmse))

    #REASONABLE NUMBER OF NODES (10)
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_rmse=[] #list to store mse of each fold
    for train, test in kfolds.split(airfoil.data, airfoil.target):
        #build neural network
        nn1 = models.Sequential()
        nn1.add(layers.Dense(10, activation='relu', input_dim = 5))
        nn1.add(layers.Dense(1))
        nn1.compile(optimizer='rmsprop', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

        #training
        n_epochs = 100
        nn1.fit(airfoil.data.iloc[train], airfoil.target.iloc[train],epochs=n_epochs)

        #testing
        s=nn1.evaluate(airfoil.data.iloc[test], airfoil.data.iloc[test])
        test_fold_rmse.append(s[1])
        print("Fold", len(test_fold_rmse),"RMSE =", s[1])

    print("\nTesting RMSE for all folds:", test_fold_rmse)
    print("\nAverage RMSE for all folds:", np.mean(test_fold_rmse))

    #TOO MANY NODES (50)
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_rmse=[] #list to store mse of each fold
    for train, test in kfolds.split(airfoil.data, airfoil.target):
        #build neural network
        nn1 = models.Sequential()
        nn1.add(layers.Dense(50, activation='relu', input_dim = 5))
        nn1.add(layers.Dense(1))
        nn1.compile(optimizer='rmsprop', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

        #training
        n_epochs = 100
        nn1.fit(airfoil.data.iloc[train], airfoil.target.iloc[train],epochs=n_epochs)

        #testing
        s=nn1.evaluate(airfoil.data.iloc[test], airfoil.data.iloc[test])
        test_fold_rmse.append(s[1])
        print("Fold", len(test_fold_rmse),"RMSE =", s[1])

    print("\nTesting RMSE for all folds:", test_fold_rmse)
    print("\nAverage RMSE for all folds:", np.mean(test_fold_rmse))


def contra():
    contra = datasets.fetch_openml(data_id=23)

    print(contra.data)
    print(contra.target)
    print("target names", contra.target_names)
    print("feature names", contra.feature_names)
    contra.data.info()

    #One Hot Encoding to do 
    ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False),[1,2,4,5,6,7,8])], remainder = "passthrough")
    new_data = ct.fit_transform(contra.data)
    #need to create df object
    new_contra_data = pd.DataFrame(new_data, columns = ct.get_feature_names_out(), index=contra.data.index)
    new_contra_data.info()

    #Convert the target into one-hot encoding
    enc = OneHotEncoder(sparse=False)
    tmp=[[x] for x in contra.target] #list of one element lists
    ohe_target=enc.fit_transform(tmp)

    #Cross Validation using SMALL NODES [15]
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_accuracy=[] #list to store test accuracy of each fold
    for train, test in kfolds.split(new_contra_data, ohe_target):
        #build neural network
        nn=models.Sequential()
        nn.add(layers.Dense(15,activation='relu', input_dim=24))
        nn.add(layers.Dense(3, activation='softmax'))
        nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        #training
        n_epochs=100
        nn.fit(new_contra_data.iloc[train], ohe_target[train], epochs=n_epochs)
        #testing
        s = nn.evaluate(new_contra_data.iloc[test], ohe_target[test])
        test_fold_accuracy.append(s[1])
        print("Fold", len(test_fold_accuracy),"Accuracy = ",s[1])
    
    print("\nTesting accuracy for all folds:", test_fold_accuracy)
    print("\nAverage testing accuracy:",np.mean(test_fold_accuracy))
    
#Cross Validation using REASONABLE NODES [30]
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_accuracy=[] #list to store test accuracy of each fold
    for train, test in kfolds.split(new_contra_data, ohe_target):
        #build neural network
        nn=models.Sequential()
        nn.add(layers.Dense(30,activation='relu', input_dim=24))
        nn.add(layers.Dense(3, activation='softmax'))
        nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        #training
        n_epochs=100
        nn.fit(new_contra_data.iloc[train], ohe_target[train], epochs=n_epochs)
        #testing
        s = nn.evaluate(new_contra_data.iloc[test], ohe_target[test])
        test_fold_accuracy.append(s[1])
        print("Fold", len(test_fold_accuracy),"Accuracy = ",s[1])
    
    print("\nTesting accuracy for all folds:", test_fold_accuracy)
    print("\nAverage testing accuracy:",np.mean(test_fold_accuracy))   

    #Cross Validation using TOO MANY NODES [45]
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_accuracy=[] #list to store test accuracy of each fold
    for train, test in kfolds.split(new_contra_data, ohe_target):
        #build neural network
        nn=models.Sequential()
        nn.add(layers.Dense(45,activation='relu', input_dim=24))
        nn.add(layers.Dense(3, activation='softmax'))
        nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        #training
        n_epochs=100
        nn.fit(new_contra_data.iloc[train], ohe_target[train], epochs=n_epochs)
        #testing
        s = nn.evaluate(new_contra_data.iloc[test], ohe_target[test])
        test_fold_accuracy.append(s[1])
        print("Fold", len(test_fold_accuracy),"Accuracy = ",s[1])
    
    print("\nTesting accuracy for all folds:", test_fold_accuracy)
    print("\nAverage testing accuracy:",np.mean(test_fold_accuracy))
    
airfoil()
contra()

