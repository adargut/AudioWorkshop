import os

for dir in os.listdir('voice_recognition/data/test'):
    for more in os.listdir('voice_recognition/data/test' + '/' + dir):
        for file in os.listdir('voice_recognition/data/test' + '/' + dir + '/' + more):
            os.remove('voice_recognition/data/test' + '/' + dir + '/' + more+'/'+file)
    for dir in os.listdir('voice_recognition/data/train'):
        cnt_in_test=0
        cnt_in_train=0
            for more in os.listdir('voice_recognition/data/train' + '/' + dir):
            print('voice_recognition/data/train' + '/' + dir + '/' + more)
           for file in os.listdir('voice_recognition/data/train' + '/' + dir + '/' + more):
           print(file)
               path_to_file='voice_recognition/data/train' + '/' + dir + '/' + more+'/'+file
   if cnt_in_train >= 20:
   print('removing')
   os.remove(path_to_file)
   elif cnt_in_train < 20 and cnt_in_test > 20:
   from pathlib import Path
   Path(path_to_file).rename('voice_recognition/data/test' + '/' + dir + '/' + more+'/'+file)
   cnt_in_train+=1
   else:
   cnt_in_test+=1