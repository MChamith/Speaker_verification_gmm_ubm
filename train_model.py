import copy
import os
import pickle

import librosa
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from extract_feature import extract_features


def map_adaptation(gmm, data, max_iterations=300, likelihood_threshold=1e-20, relevance_factor=16):
    N = data.shape[0]
    D = data.shape[1]
    K = gmm.n_components

    mu_new = np.zeros((K, D))
    n_k = np.zeros((K, 1))

    mu_k = gmm.means_
    cov_k = gmm.covariances_
    pi_k = gmm.weights_

    old_likelihood = gmm.score(data)
    new_likelihood = 0
    iterations = 0
    while (abs(old_likelihood - new_likelihood) > likelihood_threshold and iterations < max_iterations):
        iterations += 1
        old_likelihood = new_likelihood
        z_n_k = gmm.predict_proba(data)
        print('z_n_k ' + str(z_n_k.shape))
        n_k = np.sum(z_n_k, axis=0)
        n_k += 1e-10
        print('n_k ' + str(min(n_k)) )

        for i in range(K):
            temp = np.zeros((1, D))
            for n in range(N):
                temp += z_n_k[n][i] * data[n, :]
            mu_new[i] = (1 / n_k[i]) * temp

        adaptation_coefficient = n_k / (n_k + relevance_factor)
        for k in range(K):
            mu_k[k] = (adaptation_coefficient[k] * mu_new[k]) + ((1 - adaptation_coefficient[k]) * mu_k[k])
        gmm.means_ = mu_k

        log_likelihood = gmm.score(data)
        new_likelihood = log_likelihood
        print(log_likelihood)
    return gmm


speakers_path = 'development_set_enroll.txt'
dest = 'digit_models/'
file_paths = open(speakers_path, 'r')
ubm = pickle.load(open('timit_universal/ubm_512_100iter.gmm', 'rb'))
ubm_para = ubm.weights_
print(min(ubm_para))
source = 'Users/'
# ubm = open('universal_model/ubm.gmm', 'rb')
features = np.asarray(())
count = 1
model_count = 0
for wav_file in file_paths:
    path = wav_file.strip()
    # print(path)

    # read the audio
    audio, sr = librosa.load(path, 8000)
    # print('shape ' + str(audio.shape))
    # extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    if count == 34:
        print('path at 34 ' + str(path))
        ubm = copy.deepcopy(ubm)
        gmm = map_adaptation(ubm, features, max_iterations=100, relevance_factor=16)
        # gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        # gmm.fit(features)

        # dumping the trained gaussian model
        picklefile = str(model_count) + ".gmm"
        pickle.dump(gmm, open(dest + picklefile, 'wb'))
        print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
        features = np.asarray(())
        model_count +=1
        count = 0
    count = count + 1
