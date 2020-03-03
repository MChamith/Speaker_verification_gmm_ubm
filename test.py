import numpy as np
import argparse
from scipy.io.wavfile import read
from extract_feature import extract_features
from utils import calculate_likelihood
import pickle
import os

source = "Users\\"

modelpath = "speaker_models_256\\"

test_file = "development_set_test.txt"

file_paths = open(test_file, 'r')

gmm_files = [os.path.join(modelpath, fname) for fname in
             os.listdir(modelpath) if fname.endswith('.gmm')]

# Load the Gaussian gender Models
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
            in gmm_files]
def test():
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    # N = args.N
    # D = args.D
    # K = args.K
    # data = np.zeros((N,D))
    count = 0
    # map_adapted = np.load(args.map_file_name).item()
    ubm = pickle.load(open('universal_model/ubm_256_diag.gmm', 'rb'))
    for path in file_paths:
        speaker = path.split('-')[0]
        path = path.strip()
        # print(path)
        sr, audio = read(source + path)
        data = extract_features(audio, sr)
        # print(data.shape)
        N = data.shape[0]
        D = data.shape[1]
        K = ubm.n_components
        for i in range(len(models)):
            ratios = []
            map_adapted = models[i]  # checking with each model one by one
            mu_map, cov_map, pi_map = map_adapted.means_, map_adapted.covariances_, map_adapted.weights_
            mu_ubm, cov_ubm, pi_ubm = ubm.means_, ubm.covariances_, ubm.weights_
            # print('K ' + str(K))
            # print(pre_ubm.shape)
            # print(cov_ubm.shape)
            # print('mu ' + str(mu_ubm.shape))
            #             # print('cov ' + str(cov_ubm.shape))
            #             # print('pi ' + str(pi_ubm.shape))
            #             # print(N)
            #             # print(K)
            likelihood_ratio = calculate_likelihood(N, K, data, mu_map, cov_map, pi_map) - calculate_likelihood(N, K,
                                                                                                                data,
                                                                                                                mu_ubm,
                                                                                                                cov_ubm,
                                                                                                                pi_ubm)
            scores = likelihood_ratio.sum()
            # print(calculate_likelihood(N, K, data, mu_map, cov_map, pi_map))
            print(scores)
            ratios.append(scores)
        winner = np.argmin(np.array(ratios))
        print('winer number ' + str(winner))
        print(ratios)
        print("\t" + str(speaker) + "detected as - ", speakers[winner])

test()