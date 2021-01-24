import scipy.sparse as sps
import pandas as pd
import numpy as np

class DataReader(object):

    def load_urm(self):

        return pd.read_csv(filepath_or_buffer="Data/data_train.csv",
                           sep=',',
                           names = ["user_id", "item_id", "rating"],
                           header=0,
                           dtype={'row': np.int32, 'col': np.int32, 'data': np.float64})

    def load_target(self):

        df_original = pd.read_csv(filepath_or_buffer="Data/data_target_users_test.csv",
                                  sep=',', 
                                  header=0,
                                  dtype={'user_id': np.int32})

        df_original.columns = ['user']
        user_id_list = df_original['user'].values
        user_id_unique = np.unique(user_id_list)
        return user_id_unique


    def load_icm(self):
        
        df_original = pd.read_csv(filepath_or_buffer="Data/data_ICM_title_abstract.csv",
                                  sep=',',
                                  header=0,
                                  dtype={'row': np.int32, 'col': np.int32, 'data': np.float64})

        df_original.columns = ['item', 'feature', 'data']
        item_id_list = df_original['item'].values
        feature_id_list = df_original['feature'].values
        data_id_list = df_original['data'].values
        csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))
        return csr_matrix