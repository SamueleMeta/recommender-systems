from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from implicit.als import AlternatingLeastSquares
import numpy as np
import os

class ALS(BaseMatrixFactorizationRecommender):

    RECOMMENDER_NAME = "ALS"

    def fit(self, factors=1024, regularization=0.01, iterations=50, alpha=5):
        
        sparse_item_user = self.URM_train.transpose().tocsr()

        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        model = AlternatingLeastSquares(factors=factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        random_state=1234)
        
        data_confidence = (sparse_item_user * alpha).astype(np.float32)

        model.fit(data_confidence, show_progress=True)
        
        self.USER_factors = model.user_factors
        self.ITEM_factors = model.item_factors
