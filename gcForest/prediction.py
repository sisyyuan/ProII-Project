#!/bin/bash
import numpy as np
from random import sample
import matplotlib
matplotlib.use('agg')
import sys
sys.path.insert(0, "lib")
from gcforest.gcforest import GCForest
numofDrug = 708
numofProtein = 1512
positiveFeature = list()
negativeFeature = list()
negativeIndex = list()
topPrediction = list()

## Feature Fusion
drugFeature = np.loadtxt("drugFeatureNorm.txt")
proteinFeature = np.loadtxt("proteinFeatureNorm.txt")
interaction = np.loadtxt("mat_drug_protein.txt")
for i in range(numofDrug):
    for j in range(numofProtein):
        if interaction[i][j] == 1:
            positiveFeature.append(np.concatenate((drugFeature[i], proteinFeature[j]), axis=None))
        else:
            negativeFeature.append(np.concatenate((drugFeature[i], proteinFeature[j]), axis=None))
            negativeIndex.append((i,j))
topPrediction = negativeIndex
conservedNo = len(negativeIndex)
for k in np.arange(0.5, 5.5, 0.5):
    predictionNo = len(negativeIndex)
    negativeSamples = sample(negativeFeature,  int(k* len(positiveFeature)))  #
    inputData = list()
    inputData = np.concatenate((positiveFeature, negativeSamples))
    inputLable = np.repeat(np.array([1, 0]), [len(positiveFeature), len(negativeSamples)], axis=0)
    # get gcForest config
    def get_toy_config():
        config = {}
        ca_config = {}
        ca_config["random_state"] = 0
        ca_config["max_layers"] = 10
        ca_config["early_stopping_rounds"] = 3
        ca_config["n_classes"] = 2
        ca_config["estimators"] = []
        ca_config["estimators"].append(
                {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
                 "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1,"num_class":2} )
        ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
        ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
        ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
        config["cascade"] = ca_config
        return config
    X = inputData
    y = inputLable
    # Run classifier with cross-validation and plot ROC curves
    #gcForest
    config = get_toy_config()
    classifierGC = GCForest(config)
    X_train_enc = classifierGC.fit_transform(X, y)
    probas_ = classifierGC.predict_proba(np.asarray(negativeFeature))
    y_predict = classifierGC.predict(np.asarray(negativeFeature))
    #print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
    f = open("prediction.txt","a+")
    f.write("Traing on " + str(k) + " negative samples of positive ones\n")
    f.write("Probability \t")
    f.write("Index of drug_protein \n")
    f.close()
    for i in range(len(topPrediction)):
        if probas_[i][1]< 0.9:
            conservedNo -= 1
            topPrediction[i]= 0
            
    with open("prediction.txt","a+") as f:
        for i in range(len(probas_)):
            if probas_[i][1]>=0.9:
                f.write(str(probas_[i])+"\t")
                f.write(str(negativeIndex[i])+"\n")
    f = open("prediction.txt","a+")
    f.write("The prediction number of "+str(k)+" negative samples is %d" % conservedNo)
    f.close()
f = open("topPrediction.txt","a+")
f.write("The conserved top prediction number is %d" % conservedNo)
f.close()

with open("topPrediction.txt","a+") as f:
    for i in range(len(topPrediction)):
        if topPrediction[i]!= 0:
            f.write(str(topPrediction[i])+"\n")


