from geo_data_decoder import *
from eval_tools import *
import config
from read_features import *


user_feature_sequence, place_index, seg_max_record, center_location_list, useful_vec = geo_data_clean_fs()
train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec, word_index \
            = geo_dataset_train_test_text(user_feature_sequence, useful_vec, seg_max_record)