import copy
import os
import pickle
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
        n_k = np.sum(z_n_k, axis=0)

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
dest = 'speaker_models\\'
file_paths = open(speakers_path, 'r')
ubm = pickle.load(open('universal_model/ubm.gmm', 'rb'))
source = 'Users\\'
# ubm = open('universal_model/ubm.gmm', 'rb')
features = np.asarray(())
count = 1
for wav_file in file_paths:
    path = wav_file.strip()
    print(path)

    # read the audio
    sr, audio = read(source + path)

    # extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    if count == 5:
        ubm = copy.deepcopy(ubm)
        gmm = map_adaptation(ubm, features, max_iterations=10, relevance_factor=16)
        # gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        # gmm.fit(features)

        # dumping the trained gaussian model
        picklefile = path.split("-")[0] + ".gmm"
        pickle.dump(gmm, open(dest + picklefile, 'wb'))
        print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
