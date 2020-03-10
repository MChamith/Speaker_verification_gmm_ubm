import os
import shutil

for root, dirs, filenames in os.walk('/home/ubuntu/volume/libri/LibriSpeech/train-clean-100'):
    for file in filenames:
        # print('file ' + str(file) + 'count ' + str(count))
        if file.endswith('.flac'):
            # number = file.split('/')[-2]
            # if count > 2500:
            #     path = os.path.join(root, file)
            #     with open('development_set_test.txt', 'a') as fw:
            #         fw.write(path + '\n')
            path = os.path.join(root, file)
            shutil.move(path, '/home/ubuntu/volume/libri/LibriSpeech/audio/'+ str(file))


