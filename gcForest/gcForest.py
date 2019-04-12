#!/bin/bash
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from random import sample
numofDrug = 708
numofProtein = 1512
positiveFeature = list()
negativeFeature = list()

# def featureFusion(featureFile1,featureFile2,interactionFile):
feature1 = np.loadtxt("DrugFeature.txt")
feature1Norm = preprocessing.normalize(feature1, axis=0)
feature2 = np.loadtxt("ProteinFeature.txt")
feature2Norm = preprocessing.normalize(feature2, axis=0)
np.savetxt("drugFeatureNorm.txt", feature1Norm)
np.savetxt("proteinFeatureNorm.txt", feature2Norm)
interaction = np.loadtxt("mat_drug_protein.txt")
np.shape(interaction)
for i in range(numofDrug):
    for j in range(numofProtein):
        if interaction[i][j] == 1:
            positiveFeature.append(np.concatenate((feature1Norm[i], feature2Norm[j]), axis=None))
        else:
            negativeFeature.append(np.concatenate((feature1Norm[i], feature2Norm[j]), axis=None))

negativeSamples = sample(negativeFeature, 2 * len(positiveFeature))  # select 2n negative samples
inputData = list()
inputData = np.concatenate((positiveFeature, negativeSamples))
inputLable = np.repeat(np.array([1, 0]), [len(positiveFeature), len(negativeSamples)], axis=0)

X_train,X_test,y_train,y_test = train_test_split(inputData,inputLable,test_size = 0.1, random_state = 0)

import argparse
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
sys.path.insert(0, "lib")
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args

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

if __name__ == "__main__":
    args = parse_args()
    if args.model is None:
        config = get_toy_config()
    else:
        config = load_json(args.model)

    gc = GCForest(config)
    X_train_enc = gc.fit_transform(X_train, y_train)
    y_pred = gc.predict_proba(X_test)
    print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
