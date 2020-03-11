# train_models.py
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from extract_feature import extract_features
import warnings
import librosa

np_load_old = np.load
warnings.filterwarnings("ignore")

# path to training data
source = r'timit_list.txt'
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# path where training speakers will be saved
dest = "libri_universal/"
features = np.load('libri_features/feature_vector_500.npy')
print('shape ' + str(features.shape))
ubm = GaussianMixture(n_components=512, max_iter=100, covariance_type='diag', n_init=3, verbose=1)
ubm.fit(features)
# dumping the trained gaussian model
picklefile = "ubm_512_100iter.gmm"
pickle.dump(ubm, open(dest + picklefile, 'wb'))
print('modeling completed for ubm with data point = ' + str(features.shape))
