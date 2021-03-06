import os
import pickle

import librosa
import numpy as np
from scipy.io.wavfile import read
from extract_feature import extract_features
import warnings

warnings.filterwarnings("ignore")
import time

# path to training data
# source = "Users\\"

modelpath = "command_models/"

test_file = "/home/ubuntu/volume/speechcommands/testing_list.txt"

file_paths = open(test_file, 'r')

gmm_files = [os.path.join(modelpath, fname) for fname in
             os.listdir(modelpath) if fname.endswith('.gmm')]

# Load the Gaussian gender Models
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
            in gmm_files]
ubm = pickle.load(open('libri_universal/ubm_512_100iter.gmm', 'rb'))
# Read the test directory and get the list of test audio files
correct = 0
count = 0
wrong = 0
with open('result.csv', 'a') as fw:
    fw.write('Expected,Result\n')
for path in file_paths:
    speaker = path.strip('\n').split('/')[0]
    speaker = str(speaker)

    path = '/home/ubuntu/volume/speechcommands/'+str(path.strip('\n'))
    print(path)
    audio, sr = librosa.load(path)
    vector = extract_features(audio, sr)
    ubm_scores = np.array(ubm.score(vector))
    print(ubm_scores)
    ubm_loglikelihood = ubm_scores.sum()
    # gmm_file = modelpath + '\\' + str(speaker)+'.gmm'
    # print('gmm file ' + str(gmm_file))
    # gmm = pickle.load(open(gmm_file, 'rb'))
    # gmm_loglikelihood = gmm.score(vector)
    # if (gmm_loglikelihood - ubm_loglikelihood)> 5:
    #     print(speaker + ' correct, score ' + str(gmm_loglikelihood-ubm_loglikelihood))
    #     correct +=1
    # else:
    #     print(speaker + ' wrong, score ' + str(gmm_loglikelihood-ubm_loglikelihood))
    #     wrong +=1
    #
    # count +=1
    # print(ubm_loglikelihood)
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm = models[i]  # checking with each model one by one
        scores = np.array(gmm.score(vector))
        # print(scores)
        log_likelihood[i] = scores.sum()
        # print(gmm_files[i] + ' ' + str(log_likelihood[i]))
        f_score = (log_likelihood[i] - ubm_loglikelihood)
        # print('ubm score ' + str(ubm_scores) + 'map score ' + str(scores))
        print('fscore ' + str(f_score))

    winner = np.argmax(log_likelihood)
    print("\t" +str(speaker)+ "detected as - "+ str(speakers[winner]) )
    with open('result.csv', 'a') as fw:
        fw.write(str(speaker)+','+str(speakers[winner])+ '\n')
    if speaker == str(speakers[winner]):
        correct +=1
        print('correct')
    else:
        wrong +=1
        print('wrong')
    count += 1

with open('result.csv', 'a') as fw:
    fw.write('correct = '+ str(correct)+',wrong = '+ str(wrong) +'correct% =' +str(correct/count) +',wrong%= '+ str(wrong/count))

print('correct: ' + str(correct))
print('wrong ' + str(wrong))
print('total ' + str(count))
print('correct % ' + str(correct/count))
print('wrong % ' + str(wrong/count))