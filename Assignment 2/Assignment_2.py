#Cameron Lee
#Assignment 2
#Dataset ID: 42713
#***NOTE*** THIS TAKES MORE THAN AN HOUR TO RUN 

from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from scipy.stats import ttest_rel
from sklearn.metrics import mean_squared_error

def bike():
    print("***NOTE*** THIS TAKES MORE THAN AN HOUR TO RUN ")
    bike = datasets.fetch_openml(data_id=42713)
    bike.data.info()

    print(bike.data["season"].unique())
    print(bike.data["weather"].unique())

    #Remove dataframe columns corresponding to nominal features
    ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False),
                             [0,7])], remainder = "passthrough")

    #Create a new dataframe object
    new_data = ct.fit_transform(bike.data)
    ct.get_feature_names_out()
    print(ct.get_feature_names_out())

    type(new_data)
    bike_new_data = pd.DataFrame(new_data, columns = ct.get_feature_names_out(), index = bike.data.index)
    bike_new_data.info()
    print()

    #Linear Regression Base
    print("Linear Regression Base Model: Finding RMSE")
    lr = LinearRegression()
    scores = model_selection.cross_validate(lr, bike_new_data, bike.target, cv=10, scoring="neg_root_mean_squared_error")
    scores["test_score"]
    #rmseLB = RMSE Linear Base
    rmseLB = 0 - scores["test_score"]
    print(rmseLB.mean())
    print()
    
    
    #Decision Tree Regressor Base
    print("Decision Tree Regressor Base Model: Finding RMSE")
    parameters = [{"min_samples_leaf":[3,5,7,9,11,13,15]}]
    tuned_dt = GridSearchCV(DecisionTreeRegressor(), parameters, scoring = "neg_root_mean_squared_error", cv = 5)
    scores = model_selection.cross_validate(tuned_dt, bike_new_data, bike.target, cv = 10,
                                            scoring="neg_root_mean_squared_error")
    scores["test_score"]
    #rmseDTB = RMSE Decision Tree Base
    rmseDTB = 0 - scores["test_score"]
    print(rmseDTB.mean())
    print()

    #K Nearest Neighbors Regressor Base
    print("K Nearest Neighbors Regressor Base Model: Finding RMSE")
    parameters = [{"n_neighbors":[3,5,7,9,11,13,15]}]
    tuned_knn = GridSearchCV(KNeighborsRegressor(), parameters,
                             scoring = "neg_root_mean_squared_error", cv = 5)
    scores = model_selection.cross_validate(tuned_knn, bike_new_data, bike.target, cv = 10,
                                            scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseKNB = RMSE K Nearest Base
    rmseKNB = 0 - scores["test_score"]
    print(rmseKNB.mean())
    print()

    #Support-Vector Machines Regressor Base
    print("Support-Vector Machines Regressor Base Model: Finding RMSE")
    svr = SVR()
    scores = model_selection.cross_validate(svr, bike_new_data, bike.target, cv = 10,
                                            scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseSVR = Support-Vector Machines Base
    rmseSVR = 0 - scores["test_score"]
    print(rmseSVR.mean())
    print()

    #########################BAGGED REGRESSORS############################

    #Linear Bagged
    print("Linear Bagged Regressor: RMSE")
    bagged_lr = BaggingRegressor(estimator=LinearRegression())
    scores = model_selection.cross_validate(bagged_lr, bike_new_data, bike.target,
                                            cv = 10, scoring = "neg_root_mean_squared_error")
    print("I'm not really sure why this number is so big. I copied the powerpoint slide... sorry!")
    print(scores["test_score"])
    #rmseLBagged = RMSE Linear Bagged
    rmseLBagged = 0 - scores["test_score"]
    print(rmseLBagged.mean())
    print()

    #Decision Tree Bagged
    print("Decision Tree Bagged Regressor: RMSE")
    bagged_dtr = BaggingRegressor(estimator=tuned_dt)
    scores = model_selection.cross_validate(bagged_dtr, bike_new_data, bike.target,
                                            cv = 10, scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseDTBagged = RMSE Decision Tree Bagged
    rmseDTBagged = 0 - scores["test_score"]
    print(rmseDTBagged.mean())
    print()

    #KNN Bagged
    print("K Neighbors Bagged Regressor: RMSE")
    bagged_knr = BaggingRegressor(estimator=tuned_knn)
    scores = model_selection.cross_validate(bagged_knr, bike_new_data, bike.target,
                                            cv = 10, scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseKNBagged = RMSE Decision Tree Bagged
    rmseKNBagged = 0 - scores["test_score"]
    print(rmseKNBagged.mean())
    print()

    #SVR Bagged
    print("Support-Vector Machine Bagged Regressor: RMSE")
    bagged_SVM = BaggingRegressor(estimator=SVR())
    scores = model_selection.cross_validate(bagged_SVM, bike_new_data, bike.target,
                                            cv = 10, scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseSVMBagged = RMSE Support-Vector Machines Bagged
    rmseSVMBagged = 0 - scores["test_score"]
    print(rmseSVMBagged.mean())
    print()
    
    #########################BOOSTED REGRESSORS############################
    #Linear Boosted
    print("Linear Boosted Regressor: RMSE")
    boosted_lr = AdaBoostRegressor(estimator=LinearRegression())
    scores = model_selection.cross_validate(boosted_lr, bike_new_data, bike.target,
                                            cv = 10, scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseLBoosted = RMSE Linear Regression Boosted 
    rmseLBoosted= 0 - scores["test_score"]
    print(rmseLBoosted.mean())
    print()

    #Decision Tree Bagged
    print("Decision Tree Boosted Regressor: RMSE")
    boosted_dtr = AdaBoostRegressor(estimator=tuned_dt)
    scores = model_selection.cross_validate(boosted_dtr, bike_new_data, bike.target,
                                            cv = 10, scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseDTBagged = RMSE Decision Tree Bagged
    rmseDTBoosted = 0 - scores["test_score"]
    print(rmseDTBoosted.mean())
    print()

    #KNN Boosted
    print("K Neighbors Boosted Regressor: RMSE")
    boosted_knr = BaggingRegressor(estimator=tuned_knn)
    scores = model_selection.cross_validate(boosted_knr, bike_new_data, bike.target,
                                            cv = 10, scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseKNBagged = RMSE Decision Tree Bagged
    rmseKNBoosted = 0 - scores["test_score"]
    print(rmseKNBoosted.mean())
    print()
    
    #SVR Boosted
    print("Support-Vector Machine Boosted Regressor: RMSE")
    boosted_SVM = BaggingRegressor(estimator=SVR())
    scores = model_selection.cross_validate(boosted_SVM, bike_new_data, bike.target,
                                            cv = 10, scoring = "neg_root_mean_squared_error")
    scores["test_score"]
    #rmseSVMBagged = RMSE Support-Vector Machines Bagged
    rmseSVMBoosted = 0 - scores["test_score"]
    print(rmseSVMBoosted.mean())
    print()

###################VOTING REGRESSORS############################
    print("Voting Regressor: RMSE of ALL base methods")
    vr = VotingRegressor([("lr", LinearRegression()), ("svr", SVR()), ("dtc", DecisionTreeRegressor()),
                          ("knn", KNeighborsRegressor())])
    scores = model_selection.cross_validate(vr, bike_new_data, bike.target, cv = 10,
                                            scoring = "neg_root_mean_squared_error")

    scores["test_score"]
    rmseVR = 0 - scores["test_score"]
    print(rmseVR.mean())

#####################STATISTICAL SIGNIFICANCE############################

    #Subtask1 - Bagged vs Base Regressors RMSE
    print("Subtask1 - Bagged vs Base Regressors RMSE")
    print()
    
    print("Linear Regression: Bagged vs Base")
    print(ttest_rel(rmseLB, rmseLBagged))
    print()

    print("DT Regression: Bagged vs Base")
    print(ttest_rel(rmseDTB, rmseDTBagged))
    print()

    print("KN Regression: Bagged vs Base")
    print(ttest_rel(rmseKNB, rmseKNBagged))
    print()

    print("SVM Regression: Bagged vs Base")
    print(ttest_rel(rmseSVR, rmseSVMBagged))
    print()
    
    #Subtask2 - Boosted vs Base Regressors RMSE
    print("Subtask2 - Boosted vs Base Regressors RMSE")
    print()
    
    print("Linear Regression: Boosted vs Base")
    print(ttest_rel(rmseLB, rmseLBoosted))
    print()

    print("DT Regression: Boosted vs Base")
    print(ttest_rel(rmseDTB, rmseDTBoosted))
    print()

    print("KN Regression: Boosted vs Base")
    print(ttest_rel(rmseKNB, rmseKNBoosted))
    print()

    print("SVM Regression: Boosted vs Base")
    print(ttest_rel(rmseSVR, rmseSVMBoosted))
    print()

    #Subtask3 VotingRegressor vs Base Methods
    print("Subtask3 - VotingRegressor vs Base Regressors RMSE")
    print()
    
    print("Linear Regression: DT vs Base")
    print(ttest_rel(rmseLB, rmseDTB))
    print()

    print("DT Regression: VotingRegressor vs DT")
    print(ttest_rel(rmseDTB, rmseVR))
    print()

    print("KN Regression: DT vs Base")
    print(ttest_rel(rmseKNB, rmseDTB))
    print()

    print("SVM Regression: DT vs Base")
    print(ttest_rel(rmseSVR, rmseDTB))
    print()
    
bike()

















