import pickle

"""
feature extraction
"""
def FS_get_features():
    FS_data=open('./features/features&index_seg_gride_fs',"rb")
    user_feature_sequence_text, poi_index_dict, seg_max_record, center_location_list, useful_vec=pickle.load(FS_data)
    FS_data.close()
    return user_feature_sequence_text, poi_index_dict, seg_max_record, center_location_list, useful_vec


def LA_get_features():
    LA_data=open('./features/features&index_seg_gride_la',"rb")
    user_feature_sequence_text, poi_index_dict, seg_max_record, center_location_list, useful_vec=pickle.load(LA_data,encoding='iso-8859-1')
    LA_data.close()
    return user_feature_sequence_text, poi_index_dict, seg_max_record, center_location_list, useful_vec
#
# user_feature_sequence_text, poi_index_dict, seg_max_record, center_location_list, useful_vec=FS_get_features()
# print(user_feature_sequence_text)