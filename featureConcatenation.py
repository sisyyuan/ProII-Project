import numpy as np
#import pandas as pd
numofDrug = 708
numofProtein = 1512

def featureGet(numofnodes, inputfile):
    inmatrix = np.loadtxt(inputfile, skiprows=1)
    featureSelected = inmatrix[inmatrix[:,0]<numofnodes]
    featureSorted = featureSelected[featureSelected[:,0].argsort()]
    return featureSorted

def featureConcatenate(numofnodes,*files):
    flag = 0
    for file in files:
        feature = featureGet(numofnodes,file)
        feature = np.delete(feature,0,1)
        if flag == 0:
            features = feature
            flag+=1
        else:
            features = np.concatenate((features,feature),axis=1)
    return features

DrugFeatures = featureConcatenate(numofDrug,"drug_drug.emb.txt","drug_se.emb.txt","drug_disease.emb.txt")
np.savetxt("DrugFeature.txt", DrugFeatures)
ProteinFeatures = featureConcatenate(numofProtein,"protein_protein.emb.txt","protein_disease.emb.txt")
np.savetxt("ProteinFeature.txt", ProteinFeatures)
print(ProteinFeatures[249,0:99])
print(ProteinFeatures[249,100:199])


