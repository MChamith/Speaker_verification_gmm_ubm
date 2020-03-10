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
source = r'timit_list.txt'

# path where training speakers will be saved
speaker_paths = open(source, 'r')
speaker_paths = speaker_paths.readlines()

# train_file = "development_set_enroll.txt"


def create_fetaure_vector():
    features = np.asarray(())
    print('features size ' + str(features.size))
    speaker_num = 0
    for speaker_path in speaker_paths:
        speaker_path = str(speaker_path).replace('\n', '')
        print('speaker path ' + str(speaker_path))
        count = 0
        try:
            for root, dirs, filename in os.walk(str(speaker_path)):
                # print('uttername :' + str(utter_name))
                #
                # # if count == 135:
                # #     break
                # # if count > 135:
                # #     with open('development_set_test.txt', 'a') as fw :
                # #         fw.write(os.path.join(speaker_path, utter_name) +'\n')
                # #     # print('uttername break')
                # #     break
                # # for utter_file in os.listdir(os.path.join(speaker_path, utter_name)):
                # #     if count == 10:
                # #         print('innermost break')
                # #         break
                # print('utter file ' + str(utter_name))
                for file in filename:
                    print(file)
                    if file.endswith('wav'):
                        utter_path = os.path.join(root, file)

                        print('utter_path' + str(utter_path))
                        audio, sr = librosa.core.load(utter_path, 16000)
                        intervals = librosa.effects.split(audio, top_db=10)
                        vector = extract_features(audio, sr)
                        # for interval in intervals:
                        #     partial_vec = extract_features(audio, sr)
                        #     print('partial vec size  ' + str(partial_vec.size))
                        #     vector.append(partial_vec)

                        if features.size == 0:
                            features = vector
                        else:
                            print('features shape ' + str(features.shape))
                            print('vector shape ' + str(np.array(vector).shape))
                            features = np.vstack((features, vector))
                            print('feature count ' + str(features.size))
                        count += 1
        except FileNotFoundError:
            print('no file found')
            pass

    return features

# print('saving feature vector')
mfcc_features = create_fetaure_vector()
# print(mfcc_features.shape)
np.save('libri_features/feature_vector_500.npy', mfcc_features)

# audio, sr = librosa.core.load('/home/chamith/Downloads/spoken-digit-dataset/free-spoken-digit-dataset-master/recordings/1/1_jackson_49.wav', 16000)
# intervals = librosa.effects.split(audio, top_db=20)
