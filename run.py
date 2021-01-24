################################## IMPORTS ##################################

from Utils.Evaluator import EvaluatorHoldout
from Utils.DataSplitter import DataSplitter
from Utils.DataReader import DataReader
from hybrid import Hybrid
from tqdm import tqdm

################################# READ DATA #################################
reader = DataReader()
splitter = DataSplitter()
urm = reader.load_urm()
ICM = reader.load_icm()
targets = reader.load_target()

URM_train, URM_val, URM_test = splitter.split(urm, validation=0, testing=0)

####################### ISTANTIATE AND FIT THE HYBRID #######################

recommender = Hybrid(URM_train, ICM)
recommender.fit()

################################ PRODUCE CSV ################################

f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(targets):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")