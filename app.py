import os
import shutil

for root, dirs, filenames in os.walk('/home/ubuntu/volume/AudioMNIST/data'):
    for file in filenames:
        print(file)
        if file.endswith('.wav'):
            number = file.split('_')[0]
            path = os.path.join(root, file)

            shutil.move(path, '/home/ubuntu/volume/AudioMNIST/ByNumber/' + str(number) + '/' + str(file))
