# ProII-Project

The original data files are mat_*_*.txt(adjacency matrix of interaction network), *.txt(node names of network),*_dict_map.txt(dic for node names to real names fo protein and drug)


# The embedding process: embedding.py  
input: five mat_*_*.txt, 3 drug involved, and 2 protein involved  
output: five *_*.emb.txt, the first line contains number of nodes and feature dimensions(default 100), the rest lines contain the order of nodes and their embedding values


# Feature Concatenation: featureConcatenation.py  
input:five *_*.emb.txt  
output:DrugFeature.txt ProteinFeature.txt, which we need to discard the embedding results of disease/se


# Trainning dataset on SVM,KNN,Random Forest and gcForest:learningPrediction.py  
input: DrugFeature.txt ProteinFeature.txt, mat_drug_protein.txt  
number of positive samples(from mat_drug_protein.txt):1,923  
number of negative samples:1,068,573  
imbalanced dataset: negative sampling  
Training data set: positive samples and 0.5-5 fold negative samples of positive samples from random sampling
                   or equal numbers 10 times  
Training modeling: SVM,KNN,Random Forest and gcForest  
validation method: 10 fold cross validation  
output: ./gcForest/0.5-5Negative (or equalSamples)/ROC_*.pdf(for AUROC)  
        output*.txt(for accuracy)  
        gcForest works best!!  
      
      
# Prediction: prediction.py  
inputput: DrugFeature.txt ProteinFeature.txt, mat_drug_protein.txt  
Training data set: positive samples and 0.5-5 fold negative samples of positive samples   
Training method: gcForest  
Prediction data set: negativeFeatures  
output: ./gcForest/prediction.txt(for probas[1] > 0.9)  
        ./gcForest/topPrediction.txt (conserved probas[1] > 0.9 for different training data set from 0.5-5 fold of negative samples )

        
