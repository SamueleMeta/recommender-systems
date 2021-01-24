################################## IMPORTS ##################################

from Utils.SearchAbstractClass import SearchInputRecommenderArgs
from Utils.SearchBayesianSkopt import SearchBayesianSkopt
from skopt.space import Real, Integer, Categorical
from Utils.Evaluator import EvaluatorHoldout
from Utils.DataSplitter import DataSplitter
from Utils.DataReader import DataReader
import os

# Model to be tuned
from hybrid import Hybrid

################################# READ DATA #################################

reader = DataReader()
splitter = DataSplitter()

urm = reader.load_urm()
ICM = reader.load_icm()
URM_train, URM_val, URM_test = splitter.split(urm, validation=0.2, testing=0.1)

################################ EVALUATORS ##################################

evaluator_validation = EvaluatorHoldout(URM_val, [10])
evaluator_test = EvaluatorHoldout(URM_test, [10])

############################### OPTIMIZER SETUP ###############################

recommender_class = Hybrid
parameterSearch = SearchBayesianSkopt(recommender_class,
                                      evaluator_validation=evaluator_validation,
                                      evaluator_test=evaluator_test)
hyperparameters_range_dictionary = {}

'''
Insert here the hyperparameters to be tuned.
These hyperparameters should correspond to the parameters of the fit function
of the model to be tuned
'''

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}
)

output_folder_path = "result_experiments/"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

parameterSearch.search(recommender_input_args,
                       parameter_search_space = hyperparameters_range_dictionary,
                       n_cases = 200,
                       n_random_starts = 20,
                       save_model="no",
                       output_folder_path = output_folder_path,
                       output_file_name_root = recommender_class.RECOMMENDER_NAME,
                       metric_to_optimize = "MAP")