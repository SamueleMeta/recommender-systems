#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Utils.Recommender_utils import check_matrix
import similaripy


class ItemKNNCBFRecommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "ItemKNNCBFRecommender"

    def __init__(self, URM_train, ICM_train, verbose = True):
        super(ItemKNNCBFRecommender, self).__init__(URM_train, verbose = verbose)
        self.ICM_train = ICM_train


    def fit(self, topK=50, shrink=100, similarity='cosine', feature_weighting = "none"):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting == "bm25":
            self.ICM_train = similaripy.normalization.bm25(self.ICM_train)
        elif feature_weighting == "bm25plus":
            self.ICM_train = similaripy.normalization.bm25plus(self.ICM_train)
        elif feature_weighting == "tfidf":
            self.ICM_train = similaripy.normalization.tfidf(self.ICM_train)

        if similarity == "cosine":
            similarity_matrix = similaripy.cosine(self.ICM_train, k=self.topK, shrink=self.shrink, binary=False, verbose=False)
        if similarity == "s_plus":
            similarity_matrix = similaripy.s_plus(self.ICM_train, k=self.topK, shrink=self.shrink, binary = False, verbose = False)
        if similarity == "dice":
            similarity_matrix = similaripy.dice(self.ICM_train, k=self.topK, shrink=self.shrink, binary=False, verbose=False)
        if similarity == "rp3beta":
            similarity_matrix = similaripy.rp3beta(self.ICM_train, alpha=0.3, beta=0.61, k=self.topK, shrink=self.shrink, binary=False, verbose=False)
        if similarity == "p3alpha":
            similarity_matrix = similaripy.p3alpha(self.ICM_train, k=self.topK, shrink=self.shrink, binary=False, verbose=False)
        if similarity == "jaccard":
            similarity_matrix = similaripy.jaccard(self.ICM_train, k=self.topK, shrink=self.shrink, binary=False, verbose=False)

        self.W_sparse = similarity_matrix.transpose().tocsr()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

