import sys
sys.path.append('../../')

import numpy as np

from spear.labeling import PreLabels
print("jmnk")

from lfs import rules, ClassLabels
from utils import load_data_to_numpy, get_various_data
print("fown")

X, X_feats, Y = load_data_to_numpy("/raid/nlp/pranavg/pavan/azeem/RnD/train_data")

validation_size = 100
test_size = 200
L_size = 100
U_size = X.shape[0] - L_size - validation_size - test_size
# U_size = 300

print("Getting various data")
X_V,Y_V,X_feats_V,R_V, X_T,Y_T,X_feats_T,R_T, X_L,Y_L,X_feats_L,R_L, X_U,X_feats_U,R_U = get_various_data(X,Y,\
    X_feats, len(rules.get_lfs()),validation_size,test_size,L_size,U_size)
print("various data collected")

# classlabels = ClassLabels
# print(classlabels)
print("making pickle files for val")
sms_noisy_labels = PreLabels(name="oml",
                               data=X_V,
                               gold_labels=Y_V,
                               data_feats=X_feats_V,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=21)

# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms1_pickle_V.pkl')
sms_noisy_labels.generate_json('data_pipeline/sms1_json.json') #JSON
print("pickle file created")
print("making pickle files for test")

sms_noisy_labels = PreLabels(name="oml",
                               data=X_T,
                               gold_labels=Y_T,
                               data_feats=X_feats_T,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=21)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms1_pickle_T.pkl')
print("pickle file created")
print("making pickle files for labels")

sms_noisy_labels = PreLabels(name="oml",
                               data=X_L,
                               gold_labels=Y_L,
                               data_feats=X_feats_L,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=21)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms1_pickle_L.pkl')

print("pickle file created")
print("making pickle files for unlabelled")

sms_noisy_labels = PreLabels(name="oml",
                               data=X_U,
                               rules=rules,
                               data_feats=X_feats_U,
                               labels_enum=ClassLabels,
                               num_classes=21)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms1_pickle_U.pkl')

print("pickle file created")

