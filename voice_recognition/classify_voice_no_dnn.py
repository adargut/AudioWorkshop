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

MAX_TONNETZ_WEIGHT = 50
MAX_SPECTOGRAM_WEIGHT = 500
MAX_TEMPOGRAM_WEIGHT = 300
MAX_CHROMA_WEIGHT = 100
MAX_MFCC_WEIGHT = 800
MAX_PITCH_WEIGHT = 2300


class DatasetLoader:
    def __init__(self, num_threads=1, max_files_to_process_per_speaker=100, max_numbers_of_speakers=5):
        self.X = []
        self.Y = []
        self.threads = list()
        self.file_paths_to_process = list()
        self.num_threads = num_threads
        self.lock = threading.Lock()
        self.max_files_to_process_per_speaker = max_files_to_process_per_speaker
        self.progress = None
        self.max_numbers_of_speakers = max_numbers_of_speakers

    def load_dataset(self, path):
        for dir in os.listdir(path):
            processed_count = 0
            if processed_count > self.max_files_to_process_per_speaker:
                continue
            if int(dir[2:]) > 10000 + self.max_numbers_of_speakers:
                return
            if os.path.isdir(path + '/' + dir):
                for inner_dir in os.listdir(path + '/' + dir):
                    if processed_count > self.max_files_to_process_per_speaker:
                        break
                    for audio_file_to_process in os.listdir(path + '/' + dir + '/' + inner_dir):
                        full_path = path + '/' + dir + '/' + inner_dir
                        if os.path.isfile(full_path + '/' + audio_file_to_process):
                            self.file_paths_to_process.append((full_path, audio_file_to_process,))
                            processed_count += 1
                            if processed_count > self.max_files_to_process_per_speaker:
                                break

    def process_dataset(self):
        split = np.array_split(self.file_paths_to_process, self.num_threads)
        print('processing dataset with {0} threads and {1} total audio files'
              .format(self.num_threads, len(self.file_paths_to_process)))
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

    class ExtractionError(Exception):
        pass

    def process_audio_file(self, filepath, filename: str):
        label = filepath.split('/')[-2]
        try:
            duration = min(librosa.get_duration(librosa.load(filepath + '/' + filename, sr=None)[0]), 7)
            for offset in range(round(duration)):
                features = self.extract_features(filepath, filename, offset)
                self.lock.acquire()
                self.X.append(features)
                self.Y.append(label)
                self.progress.update(1)
                self.lock.release()
        except ExtractionError:
            print(f'Could not extract features from {filepath}/{filename}')
            return

    def extract_features(self, filepath, filename, offset):
        full_path = filepath + '/' + filename
        if not os.path.isfile(full_path):
            raise ExtractionError
        samples, sample_rate = librosa.load(full_path, offset=offset, sr=None, duration=1)
        tempogram = librosa.feature.tempogram(y=samples, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate)
        pitches, magnitudes = librosa.piptrack(y=samples, sr=sample_rate)
        return np.concatenate([tempogram.flatten()[:MAX_TEMPOGRAM_WEIGHT],
                               mfcc.flatten()[:MAX_MFCC_WEIGHT], magnitudes.flatten()[:MAX_PITCH_WEIGHT]])


def test_accuracy(X, Y, classifier: KNeighborsClassifier):
    err = 0
    prediction = classifier.predict(X)
    for i in range(len(prediction)):
        y, pred = Y[i], prediction[i]
        if pred != y:
            err += 1
    return 1 - (err / len(X))


CLASSIFIERS = ['knn', 'svm', 'poly_svm', 'rbf', 'dt', 'nn', 'bayes', 'ada', 'rf']


class ClassifierNotFoundException(Exception):
    pass


def get_classifier(classifier, X_train):
    if classifier == 'knn':
        return KNeighborsClassifier(n_neighbors=3)
    if classifier == 'svm':
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
    raise ClassifierNotFoundException


def main():
    train_path = 'data/wav'
    dataset_loader = DatasetLoader()
    dataset_loader.load_and_process_dataset(train_path)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_loader.X, dataset_loader.Y, test_size=0.20, random_state=42)
    best_classifier, best_acc = None, 0
    accuracies = []
    for classifier in CLASSIFIERS:
        try:
            c = get_classifier(classifier, X_train)
        except ClassifierNotFoundException:
            print(f'Invalid classifier given {classifier}')
            return
        c.fit(X_train, y_train)
        acc = test_accuracy(X_test, y_test, c)
        accuracies.append(acc)
        if acc > best_acc:
            best_classifier, best_acc = c, acc
        print(f'Achieved accuracy {round(acc, 3)} with {classifier}')

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_axes([0, 0, 1, 1])
        # ax.bar(CLASSIFIERS, accuracies)
        # plt.title('Accuracy achieved by different classifiers')
        # plt.xlabel('Classifier Name')
        # plt.ylabel('Accuracy %')
        # plt.show()

    print('Best accuracy: {:.3f}'.format(best_acc), 'achieved by: {0}'.format(best_classifier))


if __name__ == '__main__':
    main()
