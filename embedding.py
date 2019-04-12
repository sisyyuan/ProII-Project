#!/bin/bash
import networkx as nx
from node2vec import Node2Vec
import numpy as np

class Txt2Graph():
    @classmethod
    def addNodeEdge(cls,inputfile,type):
        cls.graph = nx.Graph()
        cls.info = cls.adj2info(inputfile,type)
        cls.graph.add_nodes_from(cls.info[0])
        cls.graph.add_edges_from(cls.info[1])
        return cls.graph

    @classmethod
    def adj2info(cls, inputfile,type): ##for 1 node type
        cls.inmatrix = np.loadtxt(inputfile, dtype= int)
        cls.nrow = cls.inmatrix.shape[0]
        cls.ncolum = cls.inmatrix.shape[1]
        cls.edgesList = list()
        if type ==1:
            cls.nodesList = range(cls.nrow)
            for i in range(cls.nrow):
                for j in range(cls.ncolum):
                    if cls.inmatrix[i][j] == 1:
                        cls.edgesList.append((i, j))
            return cls.nodesList, cls.edgesList  ####  # of edge is 1/2 # of 1s
        else:
            cls.nodesList = range(cls.nrow+cls.ncolum)
            for i in range(cls.nrow):
                for j in range(cls.ncolum):
                    if cls.inmatrix[i][j] == 1:
                        cls.edgesList.append((i,j+cls.nrow))
            return cls.nodesList, cls.edgesList

    def __init__(self, inputfile,type):
        self.graph = self.addNodeEdge(inputfile,type)
        self.name = inputfile[4:-4]

DrugDrugGraph = Txt2Graph("mat_drug_drug.txt",1) ##1 for one node type, 2 for two node types
DrugDiseaseGraph = Txt2Graph("mat_drug_disease.txt",2)
DrugSeGraph = Txt2Graph("mat_drug_se.txt",2)
ProteinDiseaseGraph = Txt2Graph("mat_protein_disease.txt",2)
ProteinProteinGraph = Txt2Graph("mat_protein_protein.txt",1)

graphList = [DrugDrugGraph, DrugDiseaseGraph, DrugSeGraph, ProteinDiseaseGraph, ProteinProteinGraph]

for Graph in graphList:
    node2vec = Node2Vec(Graph.graph, dimensions=100, walk_length=30, num_walks=200)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Look for most similar nodes
    #print(model.wv.most_similar('2'))   Output node names are always strings
    # FILES
    EMBEDDING_FILENAME = Graph.name + '.emb'
    EMBEDDING_MODEL_FILENAME = Graph.name + '.model'
    # Save embeddings for later use
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)
    # Save model for later use
    model.save(EMBEDDING_MODEL_FILENAME)
