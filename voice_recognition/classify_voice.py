# from __future__ import absolute_import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import librosa
import librosa.display as disp1
import os
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as disp
from sklearn.neighbors import KNeighborsClassifier
import tqdm
from sklearn.model_selection import train_test_split
import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# TODO: add bigger dataset
# TODO: more algorithms like SVM, DNN etc. done
# TODO: more features to account for done
# TODO: play around with weights done
# TODO: make a nice presentation in jupyter for muli
# TODO: understand every feature used, and comment on it.

MAX_TONNETZ_WEIGHT = 450
MAX_SPECTOGRAM_WEIGHT = 1000
MAX_TEMPOGRAM_WEIGHT = 300
MAX_CHROMA_WEIGHT = 400
MAX_MFCC_WEIGHT = 500  # most frequently used frequencies
MAX_PITCH_WEIGHT = 500  # height of vocal tone
MAX_SPEC_ROLLOF_WEIGHT = 5  # right skewedness of the waveform


class DatasetLoader:
    def __init__(self):
        self.X = []
        self.Y = []

    def load_dataset(self, path):
        for file in tqdm.tqdm(os.listdir(path), desc='Extracing audio features from data'):
            if os.path.isdir(path + '/' + file):
                for audio_file in os.listdir(path + '/' + file):
                    if file.startswith('Actor'):
                        # print(file)
                        self.process_audio_file(path + '/' + file, audio_file)

    def process_audio_file(self, filepath, filename: str):
        label = int(filepath.split('/')[-1].split('_')[-1])
        features = self.extract_features(filepath, filename)
        self.X.append(features)
        self.Y.append(label)

    def extract_features(self, filepath, filename):
        full_path = filepath + '/' + filename
        if not os.path.isfile(full_path):
            raise Exception('invalid path: ' + full_path)
        samples, sample_rate = librosa.load(full_path)
        tonnetz = librosa.feature.tonnetz(y=samples, sr=sample_rate)
        chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
        tempogram = librosa.feature.tempogram(y=samples, sr=sample_rate)
        spectogram = librosa.feature.melspectrogram(samples, sample_rate, n_fft=100)
        mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate)
        spec_rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sample_rate)
        pitches, magnitudes = librosa.piptrack(y=samples, sr=sample_rate)
        return np.concatenate([tonnetz.flatten()[:MAX_TONNETZ_WEIGHT], spectogram.flatten()[:MAX_SPECTOGRAM_WEIGHT],
                               tempogram.flatten()[:MAX_TEMPOGRAM_WEIGHT], chroma_stft.flatten()[:MAX_CHROMA_WEIGHT],
                               mfcc.flatten()[:MAX_MFCC_WEIGHT], spec_rolloff.flatten()[:MAX_SPEC_ROLLOF_WEIGHT],
                               magnitudes.flatten()[:MAX_PITCH_WEIGHT]])


def test_accuracy(X, Y, classifier: KNeighborsClassifier):
    err = 0
    prediction = classifier.predict(X)
    for i in range(len(prediction)):
        y, pred = Y[i], prediction[i]
        print('model predicted:', pred, 'actual label:', y)
        if pred != y:
            err += 1
    return 1 - (err / len(X))


def get_classifier(classifier):
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
        return MLPClassifier()
    if classifier == 'bayes':
        return GaussianNB()
    if classifier == 'ada':
        return AdaBoostClassifier()
    raise Exception('No classifier found!!')


def main():
    train_path = 'data/train_new'
    parser = argparse.ArgumentParser()
    parser.add_argument('-c')
    args = parser.parse_args()
    dataset_loader = DatasetLoader()
    dataset_loader.load_dataset(train_path)
    assert len(dataset_loader.X) == len(dataset_loader.Y)
    # X, Y = shuffle(dataset_loader.X, dataset_loader.Y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_loader.X, dataset_loader.Y, test_size=0.33, random_state=42)
    classifier = get_classifier(classifier=args.c)
    classifier.fit(X_train, y_train)
    acc = test_accuracy(X_test, y_test, classifier)
    print('done with accuracy:', acc)


if __name__ == '__main__':
    main()
