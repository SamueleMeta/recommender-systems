#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Utils.Recommender_utils import check_matrix
from Recommenders.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
import scipy.sparse as sps
import similaripy


class UserKNNCFRecommender(BaseUserSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "UserKNNCFRecommender"


    def __init__(self, URM_train, verbose = True):
        super(UserKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, topK=50, shrink=100, similarity='cosine', pre_normalization="none", post_normalization = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        #ucm = sps.load_npz("FULL_UCM.npz")
        interactions = self.URM_train

        if pre_normalization == "bm25plus":
            interactions = similaripy.normalization.bm25plus(self.URM_train, axis=1, k1=1.2, b=0.75, delta=0.85, tf_mode="raw", idf_mode="bm25", inplace=False)
        if pre_normalization == "tfidf":
            interactions = similaripy.normalization.tfidf(self.URM_train, axis=1)

        #interactions = sps.hstack((interactions, ucm))
        
        if similarity == "cosine":
            similarity_matrix = similaripy.cosine(interactions, k=self.topK, shrink=self.shrink, binary=False, verbose=False)
            similarity_matrix = similarity_matrix.transpose().tocsr()
        if similarity == "s_plus":
            similarity_matrix = similaripy.s_plus(interactions, k=self.topK, shrink=self.shrink, binary = False, verbose = False)
            similarity_matrix = similarity_matrix.transpose().tocsr()
        if similarity == "dice":
            similarity_matrix = similaripy.dice(interactions, k=self.topK, shrink=self.shrink, binary=False, verbose=False)
            similarity_matrix = similarity_matrix.transpose().tocsr()
        if similarity == "rp3beta":
            similarity_matrix = similaripy.rp3beta(interactions, alpha=0.3, beta=0.61, k=self.topK, shrink=self.shrink, binary=False, verbose=False)
            similarity_matrix = similarity_matrix.transpose().tocsr()
        if similarity == "asym":
            similarity_matrix = similaripy.asymmetric_cosine(interactions, k=self.topK, shrink=self.shrink, alpha=0.5, binary=False, verbose=False)
            similarity_matrix = similarity_matrix.transpose().tocsr()
        if similarity == "jaccard":
            similarity_matrix = similaripy.jaccard(interactions, k=self.topK, shrink=self.shrink, binary=False, verbose=False)
            similarity_matrix = similarity_matrix.transpose().tocsr()
        
        
        if post_normalization == "bm25plus_once":
            self.URM_train = similaripy.normalization.bm25plus(self.URM_train, axis=1, k1=1.2, b=0.75, delta=0.8, tf_mode='raw', idf_mode='bm25', inplace=False)
        if post_normalization == "bm25plus_twice":
            self.URM_train = similaripy.normalization.bm25plus(self.URM_train, axis=1, k1=1.2, b=0.75, delta=0.8, tf_mode='raw', idf_mode='bm25', inplace=False)
            self.URM_train = similaripy.normalization.bm25plus(self.URM_train, axis=1, k1=1.2, b=0.75, delta=0.8, tf_mode='raw', idf_mode='bm25', inplace=False)
        if post_normalization == "tfidf":
            self.URM_train = similaripy.normalization.tfidf(self.URM_train, axis=1)
        if post_normalization == "bm25":
            self.URM_train = similaripy.normalization.bm25(self.URM_train, axis=1)

        
        self.W_sparse = similarity_matrix
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
