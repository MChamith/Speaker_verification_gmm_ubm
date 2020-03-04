import os
import shutil

for i in range(10):
    count = 0
    for root, dirs, filenames in os.walk('/home/ubuntu/volume/AudioMNIST/ByNumber/'+str(i)):
        for file in filenames:
            print(file)
            if file.endswith('.wav'):
                # number = file.split('/')[-2]
                path = os.path.join(root, file)
                with open('development_set_enroll.txt', 'a') as fw:
                    fw.write(path + '\n')
