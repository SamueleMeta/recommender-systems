#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Utils.Recommender_utils import check_matrix
from Utils.DataReader import DataReader
import scipy.sparse as sps
import similaripy


class ItemKNNCFRecommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, verbose = True):
        super(ItemKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, topK=50, shrink=100, similarity='cosine', normalization = "none", feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        reader = DataReader()
        icm = reader.load_icm()

        if normalization == "bm25":
           self.URM_train = similaripy.normalization.bm25(self.URM_train, axis=1)
        if normalization == "tfidf":
            self.URM_train = similaripy.normalization.tfidf(self.URM_train, axis=1)
        if normalization == "bm25plus":
            self.URM_train = similaripy.normalization.bm25plus(self.URM_train, axis=1)

        if feature_weighting == "bm25":
            icm = similaripy.normalization.bm25(icm, axis=1)
        if feature_weighting == "tfidf":
            icm = similaripy.normalization.tfidf(icm, axis=1)
        if feature_weighting == "bm25plus":
            icm = similaripy.normalization.bm25plus(icm, axis=1)
        
        matrix = sps.hstack((self.URM_train.transpose().tocsr(), icm))
        
        if similarity == "cosine":
            similarity_matrix = similaripy.cosine(matrix, k=self.topK, shrink=self.shrink, binary=False, threshold=0)
        if similarity == "dice":
            similarity_matrix = similaripy.dice(matrix, k=self.topK, shrink=self.shrink, binary=False, threshold=0)
        if similarity == "jaccard":
            similarity_matrix = similaripy.jaccard(matrix, k=self.topK, shrink=self.shrink, binary=False, threshold=0)
        if similarity == "asym":
            similarity_matrix = similaripy.asymmetric_cosine(matrix, k=self.topK, shrink=self.shrink, binary=False, threshold=0)
        if similarity == "rp3beta":
            similarity_matrix = similaripy.rp3beta(matrix, k=self.topK, shrink=self.shrink, binary=False, threshold=0, alpha=0.3, beta=0.61)
        
        self.W_sparse = similarity_matrix
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
