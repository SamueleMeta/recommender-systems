from sklearn.model_selection import train_test_split
import scipy.sparse as sps

class DataSplitter(object):

    def split(self, data, validation, testing):
        
        num_users = 7947
        num_items = 25975
        
        (user_ids_training, user_ids_test,
        item_ids_training, item_ids_test,
        ratings_training, ratings_test) = train_test_split(data.user_id,
                                                           data.item_id,
                                                           data.rating,
                                                           test_size=testing,
                                                           shuffle=True,
                                                           random_state=9815)
        
        (user_ids_training, user_ids_validation,
        item_ids_training, item_ids_validation,
        ratings_training, ratings_validation) = train_test_split(user_ids_training,
                                                                item_ids_training,
                                                                ratings_training,
                                                                test_size=validation,
                                                                random_state=9815)
        
        urm_train = sps.csr_matrix((ratings_training, (user_ids_training, item_ids_training)), 
                                shape=(num_users, num_items))
        
        urm_validation = sps.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)), 
                                shape=(num_users, num_items))
        
        urm_test = sps.csr_matrix((ratings_test, (user_ids_test, item_ids_test)), 
                                shape=(num_users, num_items))
        
        return urm_train, urm_validation, urm_test