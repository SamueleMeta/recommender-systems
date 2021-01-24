################################# IMPORT RECOMMENDERS #################################

from Recommenders.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.RP3betaRecommender import RP3betaRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.als import ALS

################################## IMPORT LIBRARIES ##################################

from numpy import linalg as LA
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np
import similaripy

#################################### HYBRID CLASS ####################################


class Hybrid(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid"

    def __init__(self, URM_train, ICM):
        super(Hybrid, self).__init__(URM_train)
        self.ICM = ICM

    def fit(self):

        # Stack and normalize URM and ICM
        ICM = similaripy.normalization.bm25plus(self.ICM.copy())
        URM_aug = sps.vstack([self.URM_train, ICM.T])
        URM_aug2 = sps.vstack([self.URM_train, self.ICM.T])
        URM_aug2 = similaripy.normalization.bm25plus(URM_aug2)

        # Instantiate the recommenders     
        self.ItemCF1 = ItemKNNCFRecommender(self.URM_train)
        self.ItemCF2 = ItemKNNCFRecommender(self.URM_train)
        self.Beta1 = RP3betaRecommender(URM_aug)
        self.Beta2 = RP3betaRecommender(URM_aug)
        self.Als1 = ALS(URM_aug)
        self.Als2 = ALS(URM_aug2)

        # Fit the recommenders
        self.ItemCF1.fit(5000, 2000, "dice", "bm25", "bm25")
        self.ItemCF2.fit(2247, 1377, "rp3beta", "bm25plus" , "bm25")
        self.Beta1.fit(alpha=0.534650819474422, beta=0.45839478656885524, min_rating=0, topK=1883, implicit=True, normalize_similarity=False)
        self.Beta2.fit(alpha=0.6141574280038155, beta=0.3138532455756053, min_rating=0, topK=1161, implicit=True, normalize_similarity=False)
        self.Als1.fit(factors=550, regularization=0.001, iterations=60, alpha=65)
        self.Als2.fit(factors=450, regularization=0.001, iterations=15, alpha=4)
        
    
    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = np.empty([len(user_id_array), 25975])
        for i in tqdm(range(len(user_id_array))):

            interactions = len(self.URM_train[user_id_array[i],:].indices)

            if interactions < 8: 
                w1 = self.Als2._compute_item_score(user_id_array[i], items_to_compute)  
                w1 /= LA.norm(w1, 2)
                w2 = self.ItemCF2._compute_item_score(user_id_array[i], items_to_compute)  
                w2 /= LA.norm(w2, 2)
                w = w1 + w2 
                item_weights[i,:] = w 
            elif interactions > 7 and interactions < 18: 
                w1 = self.ItemCF2._compute_item_score(user_id_array[i], items_to_compute) 
                w2 = self.Beta2._compute_item_score(user_id_array[i], items_to_compute) 
                w3 = w1 * 2.7341732749480965 + w2 * 1.7291203225266563 
                w3 /= LA.norm(w3, 2)
                w4 = self.Als1._compute_item_score(user_id_array[i], items_to_compute) 
                w4 /= LA.norm(w4, 2)
                w = w3 + w4
                item_weights[i,:] = w
            else:
                w1 = self.Beta1._compute_item_score(user_id_array[i], items_to_compute) 
                w2 = self.ItemCF1._compute_item_score(user_id_array[i], items_to_compute) 
                w3 = w1 * 3.1354787809646 + w2 * 0.6847368170848224 
                w3 /= LA.norm(w3, 2)
                w4 = self.Als1._compute_item_score(user_id_array[i], items_to_compute) 
                w4 /= LA.norm(w4, 2)
                w = w3 + w4 * 1.75
                item_weights[i,:] = w 

        return item_weights
    
    
    