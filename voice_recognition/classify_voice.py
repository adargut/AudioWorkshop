import librosa
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import threading
import time
import tqdm
from sklearn import preprocessing


# TODO: add bigger dataset
# TODO: more algorithms like SVM, DNN etc. done
# TODO: more features to account for done
# TODO: play around with weights done
# TODO: make a nice presentation in jupyter for muli
# TODO: understand every feature used, and comment on it.
# TODO save feature extraction to file
# TODO make data processing multi threaded

# TODO: explain plots
# TODO: improve accuracy
# TODO: create test train deterministic to split equally


MAX_TONNETZ_WEIGHT = 500
MAX_SPECTOGRAM_WEIGHT = 1
MAX_TEMPOGRAM_WEIGHT = 1000
MAX_CHROMA_WEIGHT = 1
MAX_MFCC_WEIGHT = 2300
MAX_PITCH_WEIGHT = 1000


class DatasetLoader:
    def __init__(self, num_threads=30, max_files_to_process_per_speaker=2, min_audio_file_duration=8,
                 max_numbers_of_speakers=50):
        self.X = []
        self.Y = []
        self.threads = list()
        self.file_paths_to_process = list()
        self.num_threads = num_threads
        self.lock = threading.Lock()
        self.max_files_to_process_per_speaker = max_files_to_process_per_speaker
        self.progress = None
        self.min_audio_file_duration = min_audio_file_duration
        self.max_numbers_of_speakers = max_numbers_of_speakers

    def load_dataset(self, path):
        for dir in os.listdir(path):
            processed_count = 0
            if processed_count > self.max_files_to_process_per_speaker:
                continue
            # if int(dir[2:]) > 10000 + self.max_numbers_of_speakers:
            #     return
            if os.path.isdir(path + '/' + dir):
                for inner_dir in os.listdir(path + '/' + dir):
                    if processed_count > self.max_files_to_process_per_speaker:
                        break
                    for audio_file_to_process in os.listdir(path + '/' + dir + '/' + inner_dir):
                        full_path = path + '/' + dir + '/' + inner_dir
                        # if librosa.get_duration(librosa.load(full_path + '/' + audio_file_to_process, sr=None)[0]) < self.min_audio_file_duration:
                        #     continue
                        if os.path.isfile(full_path + '/' + audio_file_to_process):
                            self.file_paths_to_process.append((full_path, audio_file_to_process,))
                            processed_count += 1
                            if processed_count > self.max_files_to_process_per_speaker:
                                break

    def process_dataset(self):
        split = np.array_split(self.file_paths_to_process, self.num_threads)
        print('processing dataset with {0} threads and {1} total audio files'.format(self.num_threads,
                                                                                     len(self.file_paths_to_process)))
        self.progress = tqdm.tqdm(total=len(self.file_paths_to_process), desc='Extracting features from audio')

        with self.progress:
            for i in range(self.num_threads):
                t = threading.Thread(target=self.process_audio_files_array, args=([split[i]]))
                self.threads.append(t)
                t.start()
            for t in self.threads:
                t.join()

    def load_and_process_dataset(self, path):
        self.load_dataset(path)
        start = time.time()
        self.process_dataset()
        end = time.time()
        print('finished processing dataset in {:.3f} ms'.format(end - start))
        return self

    def process_audio_files_array(self, arr):
        for filepath, filename in arr:
            self.process_audio_file(filepath, filename)

    def process_audio_file(self, filepath, filename: str):
        label = filepath.split('/')[-2]
        features = self.extract_features(filepath, filename)
        self.lock.acquire()
        self.X.append(features)
        self.Y.append(label)
        self.progress.update(1)
        self.lock.release()

    def extract_features(self, filepath, filename):
        full_path = filepath + '/' + filename
        if not os.path.isfile(full_path):
            raise Exception('invalid path: ' + full_path)
        samples, sample_rate = librosa.load(full_path, sr=None, duration=4)
        chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
        tempogram = librosa.feature.tempogram(y=samples, sr=sample_rate)
        # spectogram = librosa.feature.melspectrogram(samples, sample_rate)
        # tonnetz = librosa.feature.tonnetz(y=samples, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate)
        pitches, magnitudes = librosa.piptrack(y=samples, sr=sample_rate)
        # concat = np.concatenate([tonnetz.flatten()[:MAX_TONNETZ_WEIGHT], spectogram.flatten()[:MAX_SPECTOGRAM_WEIGHT],
        #                        tempogram.flatten()[:MAX_TEMPOGRAM_WEIGHT],
        #                        chroma_stft.flatten()[:MAX_CHROMA_WEIGHT],
        #                        mfcc.flatten()[:MAX_MFCC_WEIGHT], magnitudes.flatten()[:MAX_PITCH_WEIGHT]])
        # return concat
        # return preprocessing.normalize([concat])[0]
        # return preprocessing.normalize([mfcc.flatten()])
        # return mfcc.flatten()[:MAX_MFCC_WEIGHT]
        pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
        return pad2d(mfcc, 40)
        # return np.concatenate([mfcc.flatten()[:MAX_MFCC_WEIGHT], tempogram.flatten()[:MAX_TEMPOGRAM_WEIGHT], chroma_stft.flatten()[:MAX_CHROMA_WEIGHT]])


def test_accuracy(X, Y, classifier: KNeighborsClassifier):
    err = 0
    prediction = classifier.predict(X)
    for i in range(len(prediction)):
        y, pred = Y[i], prediction[i]
        print('model predicted:', pred, 'actual label:', y)
        if pred != y:
            err += 1
    return 1 - (err / len(X))


CLASSIFIERS = ['knn', 'linear_svm', 'poly_svm', 'rbf', 'dt', 'nn', 'bayes', 'ada', 'rf', 'sknn']


def get_classifier(classifier, X_train):
    print('classifier:', classifier)
    if classifier == 'knn':
        return KNeighborsClassifier(n_neighbors=1)
    if classifier == 'linear_svm':
        return SVC(kernel='linear')
    if classifier == 'poly_svm':
        return SVC(kernel='poly')
    if classifier == 'rbf':
        return SVC(kernel='rbf')
    if classifier == 'dt':
        return DecisionTreeClassifier()
    if classifier == 'nn':
        return MLPClassifier(hidden_layer_sizes=(500,), alpha=0.1, learning_rate_init=0.1)
    if classifier == 'bayes':
        return GaussianNB()
    if classifier == 'ada':
        return AdaBoostClassifier()
    if classifier == 'rf':
        return RandomForestClassifier()
    if classifier == 'sknn':
        import tensorflow as tf
        ip = tf.keras.layers.Input(shape=X_train[0].shape)
        m = tf.keras.layers.Conv2D(256, kernel_size=(2,2),activation='relu')(ip)
        m = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(m)
        m = tf.keras.layers.BatchNormalization()(m)
        m = tf.keras.layers.Dropout(0.2)(m)
        m = tf.keras.layers.Flatten()(m)
        m = tf.keras.layers.Dense(64, activation='relu')(m)
        op = tf.keras.layers.Dense(12, activation='softmax')(m)
        model = tf.keras.Model(inputs=ip, outputs=op)
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
    raise Exception('No classifier found!!')


def main():
    train_path = 'data/wav'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c')
    # args = parser.parse_args()
    # classifier = get_classifier(classifier=args.c)
    # dataset_loader = DatasetLoader()
    # dataset_loader.load_and_process_dataset(train_path)
    train_dataset_loader = DatasetLoader()
    train_path = 'data/train'
    train_dataset_loader.load_and_process_dataset(train_path)
    test_dataset_loader = DatasetLoader()
    test_path = 'data/test'
    test_dataset_loader.load_and_process_dataset(test_path)
    # assert len(dataset_loader.X) == len(dataset_loader.Y)
    # X, Y = shuffle(dataset_loader.X, dataset_loader.Y, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     dataset_loader.X, dataset_loader.Y, test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_dataset_loader.X, test_dataset_loader.X, \
                                       train_dataset_loader.Y, test_dataset_loader.Y
    # X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
    # X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
    best_classifier, best_acc = None, 0
    for classifier in ['sknn']:#CLASSIFIERS:
        X_train = np.array(X_train)
        for i in range(len(y_train)):
            y_train[i] = int(y_train[i][5:]) - 29
        for i in range(len(y_test)):
            y_test[i] = int(y_test[i][5:]) - 29
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        X_test = np.array(X_test)
        train_X_ex = np.expand_dims(X_train, -1)
        test_X_ex = np.expand_dims(X_test, -1)
        c = get_classifier(classifier, train_X_ex)
        from tensorflow.keras.utils import to_categorical
        y_train = to_categorical(np.array(y_train))
        X_train = np.stack([i.tolist() for i in X_train])
        c.fit(train_X_ex, y_train, epochs=100,
          batch_size=32,
          validation_split=0.1)
        acc = test_accuracy(X_test, y_test, c)
        print('Accuracy achieved {:.3f}'.format(acc), 'with classifier {0}'.format(classifier))
        if acc > best_acc:
            best_classifier, best_acc = c, acc
    # classifier.fit(X_train, y_train)
    # acc = test_accuracy(X_test, y_test, classifier)
    print('Best accuracy: {:.3f}'.format(best_acc), 'achieved by: {0}'.format(best_classifier))


if __name__ == '__main__':
    main()