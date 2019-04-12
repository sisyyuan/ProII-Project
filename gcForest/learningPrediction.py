#!/bin/bash
import statistics
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from random import sample
from scipy import interp
import matplotlib
matplotlib.use('agg')
import pylab as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0, "lib")
from gcforest.gcforest import GCForest
import math
numofDrug = 708
numofProtein = 1512
positiveFeature = list()
negativeFeature = list()

## Feature Fusion
feature1 = np.loadtxt("DrugFeature.txt")
feature1Norm = preprocessing.normalize(feature1, axis=0)
feature2 = np.loadtxt("ProteinFeature.txt")
feature2Norm = preprocessing.normalize(feature2, axis=0)
#np.savetxt("drugFeatureNorm.txt", feature1Norm)
#np.savetxt("proteinFeatureNorm.txt", feature2Norm)
interaction = np.loadtxt("mat_drug_protein.txt")
np.shape(interaction)
for i in range(numofDrug):
    for j in range(numofProtein):
        if interaction[i][j] == 1:
            positiveFeature.append(np.concatenate((feature1Norm[i], feature2Norm[j]), axis=None))
        else:
            negativeFeature.append(np.concatenate((feature1Norm[i], feature2Norm[j]), axis=None))
for k in range(10):
    negativeSamples = sample(negativeFeature,  len(positiveFeature))  # select 2n negative samples
    inputData = list()
    inputData = np.concatenate((positiveFeature, negativeSamples))
    inputLable = np.repeat(np.array([1, 0]), [len(positiveFeature), len(negativeSamples)], axis=0)

    # get gcForest config
    def get_toy_config():
        config = {}
        ca_config = {}
        ca_config["random_state"] = 0
        ca_config["max_layers"] = 100
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
    random_state = np.random.RandomState(0)
    cv = StratifiedKFold(n_splits=10)
    classifierSVM = svm.SVC(kernel='linear', probability=True,random_state=random_state)
    classifierKNN = KNeighborsClassifier(n_neighbors=8)
    classifierRF = RandomForestClassifier(n_estimators = 10, random_state = random_state)
    #gcForest
    config = get_toy_config()
    classifierGC = GCForest(config)
    classifiers =[classifierSVM, classifierKNN, classifierRF, classifierGC]
    for classifier in classifiers:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 0
        acc = []
        for train, test in cv.split(X, y):
            if classifier == classifierGC:
                # gcForest
                X_train_enc = classifierGC.fit_transform(X[train], y[train])
                probas_ = classifierGC.predict_proba(X[test])
                y_predict = classifierGC.predict(X[test])
                #print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
            else:
                probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
                y_predict = classifier.fit(X[train], y[train]).predict(X[test])
            acc.append(accuracy_score(y_predict,y[test]))
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC: '+ classifier.__class__.__name__ + str(k))
        art = []
        lgd = plt.legend(bbox_to_anchor=(1, 0), loc='lower left', ncol=1)
        #plt.tight_layout()
        plt.savefig("./equalSamples/ROC_" + classifier.__class__.__name__+ str(k) + ".pdf",additional_artists = art, bbox_inches = "tight")
        plt.close()
        print(classifier.__class__.__name__+ str(k) + "Accuracy:%0.2f(+/- %0.2f)" % (np.mean(acc),statistics.stdev(acc)*2))
        file = open("./equalSamples/output_1.txt", "a+")
        file.write("\n\n" + classifier.__class__.__name__+ str(k)+" Accuracies:\n\n")
        file.write(str(acc))
        file.write("\n\n" + classifier.__class__.__name__+ str(k) + " Accuracy:%0.2f(+/- %0.2f)" % (np.mean(acc),statistics.stdev(acc)*2))
        file.close()

