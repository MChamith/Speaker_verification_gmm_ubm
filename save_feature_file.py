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
speaker_paths = open(source, 'r')
# train_file = "development_set_enroll.txt"


def create_fetaure_vector():
    features = np.asarray(())
    speaker_num = 0
    for speaker_path in speaker_paths:
        speaker_path = str(speaker_path).replace('\n', '')
        count = 0
        try:
            for utter_name in os.listdir(speaker_path):
                print('uttername :' + str(utter_name))
                if count == 10:
                    print('uttername break')
                    break
                for utter_file in os.listdir(os.path.join(speaker_path, utter_name)):
                    if count == 10:
                        print('innermost break')
                        break
                    print('utter file ' + str(utter_file))
                    utter_path = os.path.join(os.path.join(speaker_path, utter_name),
                                              utter_file)  # path of each utterance
                    print('utter_path' + str(utter_path))
                    audio, sr = librosa.core.load(utter_path)
                    vector = extract_features(audio, sr)
                    if features.size == 0:
                        features = vector
                    else:
                        features = np.vstack((features, vector))
                        print('feature count ' + str(features.size))
                    count += 1
        except FileNotFoundError:
            pass
        speaker_num +=1
        if speaker_num == 500:
            return features

print('saving feature vector')
features = create_fetaure_vector()
np.save('features/feature_vector_500.npy', features)
