# train_models.py
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from extract_feature import extract_features
import warnings
import librosa

warnings.filterwarnings("ignore")

# path to training data
source = r'voice_list.txt'

# path where training speakers will be saved
dest = "universal_model\\"
features = np.load('features/feature_vector_500.npy')
ubm = GaussianMixture(n_components=512, max_iter=100, covariance_type='diag', n_init=3, verbose=1)
ubm.fit(features)
# dumping the trained gaussian model
picklefile = "ubm_512_100iter.gmm"
pickle.dump(ubm, open(dest + picklefile, 'wb'))
print('modeling completed for ubm with data point = ' + str(features.shape))
