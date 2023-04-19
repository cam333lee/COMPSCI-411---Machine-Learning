#CAMERON LEE
#2/22/2023
#ASSIGNMENT 1
#-------------------------------------------------------------------------------------------------------------

#Datasets Used:
#1.) Internet-Advertisement
#1.) https://www.openml.org/search?type=data&status=active&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_1000_10000&sort=qualities.NumberOfNumericFeatures&id=40978

#2.) Bioresponse
#2.) https://www.openml.org/search?type=data&status=active&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_1000_10000&sort=qualities.NumberOfNumericFeatures&id=4134
#-------------------------------------------------------------------------------------------------------------

#LOADING THE DATA
from sklearn import datasets
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt

def subtask1():
    print("Subtask1: Internet-Advertisement")
    #Get Internet-Advertisement data from source
    dia1 = datasets.fetch_openml(data_id=40978)

    #Display data
    print("Displaying feature names of Internet-Advertisement: ")
    print(dia1.feature_names)

    print("Displaying all data from Internet-Advertisement: ")
    print(dia1.data)

    print("Displaying target from Internet-Advertisement: ")
    print(dia1.target)

    #Creating empty list to hold all of the decision trees
    decisionTree1List = []
    for x in range(1,6):
        decisionTree1List.insert(x -1, tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf = x))

    print("")
    print("Looping through all decision trees in array")
    [print(x) for x in decisionTree1List]

    #Fitting each of the trees:
    print("")
    print("Fitting all of the trees based on the training data: ")

    for x in range(5):
        print("Decision Tree with min_samples_leaf = " + str(x + 1)) 
        decisionTree1List[x].fit(dia1.data, dia1.target)
        print(tree.export_text(decisionTree1List[x]))

    #MAKING PREDICTIONS
    predictionSubtask1List = []
    for x in range(1, 6):
        #This was tested on the training data 
        predictionSubtask1List.insert(x - 1, decisionTree1List[x-1].predict(dia1.data))
        print("Predicted values with min_samples_leaf = " + str(x))
        print(predictionSubtask1List[x -1])

    #predictionSubtask1List[0] = decisionTree1List[0].predict(dia1.

    #EVALUATING
    PredictionList1Probs = []
    for x in range (1, 6):
        #Still on training data
        PredictionList1Probs.insert(x - 1, decisionTree1List[x-1].predict_proba(dia1.data))
        print("These are the predicted probabilities with min_samples_leaf numbers = " + str(x))
        print(PredictionList1Probs[x-1])        
                                    
    #ATTAINING ROC SCORE FROM PREDICTION SCORES (WITHOUT 10-FOLD CROSS VALIDATION)
    PredictionROCAUCNo10 = []
    for x in range (1, 6):
        #Still on training data
        PredictionROCAUCNo10.insert(x-1, metrics.roc_auc_score(dia1.target, PredictionList1Probs[x-1][:,1]))
        print("The ROC AUC scores leaf sample = " + str(x) + " (without 10-Fold Cross Validation)")
        print(PredictionROCAUCNo10[x-1])

    #MEASURING TEST ROC_AUC SCORES ON 10-FOLD CROSS-VALIDATION
    CVRocAucScores1 = []
    CVRocAUCMeanTestScores1 = []
    CVRocAUCMeanTrainScores1 = []
    
    for x in range (1, 6):
        CVRocAucScores1.insert(x-1, model_selection.cross_validate(decisionTree1List[x-1], dia1.data, dia1.target, scoring = "roc_auc", cv = 10, return_train_score = True))
        #print(CVRocAucTestScores1[x-1]["test_score"])
        print("Decision Tree #" + str(x) + ":")
        #print(CVRocAucTestScores1[x-1]) //Prints out the right scores
        print(CVRocAucScores1[x-1]["test_score"])

        #Inserting into new array to hold of all of the test scores 
        CVRocAUCMeanTestScores1.insert(x-1,CVRocAucScores1[x-1]["test_score"].mean())
        print("Decision Tree #" + str(x) + " Mean ROC AUC Test Score of all 10 folds:")
        print(CVRocAUCMeanTestScores1[x-1])

        #Inserting into new array to old all of the training scores
        CVRocAUCMeanTrainScores1.insert(x-1,CVRocAucScores1[x-1]["train_score"].mean())
        print("Decision Tree #" + str(x) + " Mean ROC AUC Training Score of all 10 folds:")
        print(CVRocAUCMeanTrainScores1[x-1])
        print("")
        
    #GRAPHING THE ROC TEST MEAN ROC CURVES PER FOLD 
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[0]["test_score"], label = "1 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[1]["test_score"], label = "2 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[2]["test_score"], label = "3 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[3]["test_score"], label = "4 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[4]["test_score"], label = "5 Min Sample Leafs")
    plt.xlabel("N fold")
    plt.ylabel('Test Score')
    plt.legend()
    plt.title("ROC AUC Test Scores Curve Per Fold: Internet-Advertisement") 
    plt.show()

    #GRAPHING THE ROC TRAIN MEAN ROC CURVES PER FOLD
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[0]["train_score"], label = "1 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[1]["train_score"], label = "2 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[2]["train_score"], label = "3 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[3]["train_score"], label = "4 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores1[4]["train_score"], label = "5 Min Sample Leafs")
    plt.xlabel("N fold")
    plt.ylabel('Train Score')
    plt.legend()
    plt.title("ROC AUC Train Scores Curve Per Fold: Internet-Advertisement") 
    plt.show()

    #GRAPHING THE ROC TEST MEAN PER LEAF
    plt.plot([1, 2, 3, 4, 5], [CVRocAUCMeanTestScores1[0], CVRocAUCMeanTestScores1[1], CVRocAUCMeanTestScores1[2], CVRocAUCMeanTestScores1[3], CVRocAUCMeanTestScores1[4]])
    plt.xlabel("# Of Min_Samples Leaf")
    plt.ylabel("Mean ROC_AUC Score Per Min # Of Sample Leaves")
    plt.title("Mean ROC_AUC Test Scores Per Leaf: Internet-Advertisement")
    plt.show()
    

    #Bar graph of min_samples_parameter and the mean roc_auc score
    #data1 = {'1':CVRocAUCMeanTestScores1[0], '2':CVRocAUCMeanTestScores1[1], '3':CVRocAUCMeanTestScores1[2], '4':CVRocAUCMeanTestScores1[3], '5':CVRocAUCMeanTestScores1[4]}

    #samples1 = list(data1.keys())
    #values1 = list(data1.values())
    #fig = plt.figure(figsize = (10, 5))
    x1 = ["1", "2", "3", "4", "5"]
    y1 = [CVRocAUCMeanTestScores1[0], CVRocAUCMeanTestScores1[1], CVRocAUCMeanTestScores1[2], CVRocAUCMeanTestScores1[3], CVRocAUCMeanTestScores1[4]]

    
    plt.bar(x1, y1, color = 'lightblue')

    plt.xlabel("# Of Min_Samples Leaf")
    plt.ylabel("Mean ROC_AUC Score out of the 10 Folds")
    plt.title("Mean ROC_AUC Test Scores Per Leaf: Internet-Advertisement")
    plt.show()


    print("Subtask1: Bioresponse")
    #Get Internet-Advertisement data from source
    dia2 = datasets.fetch_openml(data_id=4134)

    #Display data
    print("Displaying feature names of Bioresponse: ")
    print(dia2.feature_names)

    print("Displaying all data from Bioresponse: ")
    print(dia2.data)

    print("Displaying target from Bioresponse: ")
    print(dia2.target)

    #Creating empty list to hold all of the decision trees
    decisionTree2List = []
    for x in range(1,6):
        decisionTree2List.insert(x -1, tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf = x))

    print("")
    print("Looping through all decision trees in array")
    [print(x) for x in decisionTree2List]

    #Fitting each of the trees:
    print("")
    print("Fitting all of the trees based on the training data: ")

    for x in range(5):
        print("Decision Tree with min_samples_leaf = " + str(x + 1)) 
        decisionTree2List[x].fit(dia2.data, dia2.target)
        print(tree.export_text(decisionTree2List[x]))

    #MAKING PREDICTIONS
    predictionSubtask2List = []
    for x in range(1, 6):
        #This was tested on the training data 
        predictionSubtask2List.insert(x - 1, decisionTree2List[x-1].predict(dia2.data))
        print("Predicted values with min_samples_leaf = " + str(x))
        print(predictionSubtask2List[x -1])

    #EVALUATING
    PredictionList2Probs = []
    for x in range (1, 6):
        #Still on training data
        PredictionList2Probs.insert(x - 1, decisionTree2List[x-1].predict_proba(dia2.data))
        print("These are the predicted probabilities with min_samples_leaf numbers = " + str(x))
        print(PredictionList2Probs[x-1])        
                                    
    #ATTAINING ROC SCORE FROM PREDICTION SCORES (WITHOUT 10-FOLD CROSS VALIDATION)
    PredictionROCAUCNo102 = []
    for x in range (1, 6):
        #Still on training data
        PredictionROCAUCNo102.insert(x-1, metrics.roc_auc_score(dia2.target, PredictionList2Probs[x-1][:,1]))
        print("The ROC AUC scores leaf sample = " + str(x) + " (Without 10-Fold Cross Validation)")
        print(PredictionROCAUCNo102[x-1])

    #MEASURING TEST ROC_AUC SCORES ON 10-FOLD CROSS-VALIDATION
    CVRocAucScores2 = []
    CVRocAUCMeanTestScores2 = []
    CVRocAUCMeanTrainScores2 = []

    for x in range (1, 6):
        CVRocAucScores2.insert(x-1, model_selection.cross_validate(decisionTree2List[x-1], dia2.data, dia2.target, scoring = "roc_auc", cv = 10, return_train_score = True))
        #print(CVRocAucTestScores1[x-1]["test_score"])
        print("Decision Tree #" + str(x) + ":")
        #print(CVRocAucScores2[x-1]) //Prints out the right scores
        print(CVRocAucScores2[x-1]["test_score"])

        #Inserting into new array to hold of all of the test scores 
        CVRocAUCMeanTestScores2.insert(x-1,CVRocAucScores2[x-1]["test_score"].mean())
        print("Decision Tree #" + str(x) + " Mean ROC AUC Test Score of all 10 folds:")
        print(CVRocAUCMeanTestScores2[x-1])

        #Inserting into new array to old all of the training scores
        CVRocAUCMeanTrainScores2.insert(x-1,CVRocAucScores2[x-1]["train_score"].mean())
        print("Decision Tree #" + str(x) + " Mean ROC AUC Training Score of all 10 folds:")
        print(CVRocAUCMeanTrainScores2[x-1])
        print("")
        
    #GRAPHING THE ROC TEST MEAN ROC CURVES 
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[0]["test_score"], label = "1 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[1]["test_score"], label = "2 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[2]["test_score"], label = "3 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[3]["test_score"], label = "4 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[4]["test_score"], label = "5 Min Sample Leafs")
    plt.xlabel("N fold")
    plt.ylabel('Test Score')
    plt.legend()
    plt.title("ROC AUC Test Scores Curve Per Fold: Bioresponse") 
    plt.show()

    #GRAPHING THE ROC TRAIN MEAN ROC CURVES 
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[0]["train_score"], label = "1 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[1]["train_score"], label = "2 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[2]["train_score"], label = "3 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[3]["train_score"], label = "4 Min Sample Leafs")
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], CVRocAucScores2[4]["train_score"], label = "5 Min Sample Leafs")
    plt.xlabel("N fold")
    plt.ylabel('Train Score')
    plt.legend()
    plt.title("ROC AUC Train Scores Curve Per Fold: Bioresponse") 
    plt.show()

    #GRAPHING THE ROC TEST MEAN PER LEAF
    plt.plot([1, 2, 3, 4, 5], [CVRocAUCMeanTestScores2[0], CVRocAUCMeanTestScores2[1], CVRocAUCMeanTestScores2[2], CVRocAUCMeanTestScores2[3], CVRocAUCMeanTestScores2[4]])
    plt.xlabel("# Of Min_Samples Leaf")
    plt.ylabel("Mean ROC_AUC Score Per Min # Of Sample Leaves")
    plt.title("Mean ROC_AUC Test Scores Per Leaf: Bioresponse")
    plt.show()

    #Bar Graph
    x2 = ["1", "2", "3", "4", "5"]
    y2 = [CVRocAUCMeanTestScores2[0], CVRocAUCMeanTestScores2[1], CVRocAUCMeanTestScores2[2], CVRocAUCMeanTestScores2[3], CVRocAUCMeanTestScores2[4]]

    
    plt.bar(x2, y2, color = 'lightblue')

    plt.xlabel("# Of Min_Samples Leaf")
    plt.ylabel("Mean ROC_AUC Score out of the 10 Folds")
    plt.title("Mean ROC_AUC Test Scores Per Leaf: Bioresponse")
    plt.show()
            

    
def subtask2():
    print("Internet-Advertisement Best Parameter")
    dia1 = datasets.fetch_openml(data_id=40978)

    parameters1 = [{"min_samples_leaf":[2, 4, 6, 8, 10]}]

    print("After parameters")

    dtc1 = tree.DecisionTreeClassifier()
    print("After dtc1")
    tuned_dtc1 = model_selection.GridSearchCV(dtc1, parameters1, scoring = "roc_auc", cv = 5)
    print("After grid search")
    cv1 = model_selection.cross_validate(tuned_dtc1, dia1.data, dia1.target, scoring = "roc_auc", cv = 10, return_train_score = True)
    print("The Test Score Mean of the Parameter Tuned Data is: " + str(cv1["test_score"].mean()))
    print("The Training Score Mean of the Parameter Tuned Data is: " + str(cv1["train_score"].mean()))
    
    #Best Parameter For Training
    tuned_dtc1.fit(dia1.data, dia1.target)
    print("The best parameter is: " + str(tuned_dtc1.best_params_))


    print("Bioresponse Best Parameter")
    dia2 = datasets.fetch_openml(data_id=4134)

    parameters2 = [{"min_samples_leaf":[2, 4, 6, 8, 10]}]

    print("After parameters")

    dtc2 = tree.DecisionTreeClassifier()
    print("After dtc1")
    tuned_dtc2 = model_selection.GridSearchCV(dtc2, parameters2, scoring = "roc_auc", cv = 5)
    print("After grid search")
    cv2 = model_selection.cross_validate(tuned_dtc2, dia2.data, dia2.target, scoring = "roc_auc", cv = 10, return_train_score = True)
    print("The Test Score Mean of the Parameter Tuned Data is: " + str(cv2["test_score"].mean()))
    print("The Training Score Mean of the Parameter Tuned Data is: " + str(cv2["train_score"].mean()))
    
    #Best Parameter For Training
    tuned_dtc2.fit(dia2.data, dia2.target)
    print("The best parameter is: " + str(tuned_dtc2.best_params_))

subtask1() 
subtask2()



    
