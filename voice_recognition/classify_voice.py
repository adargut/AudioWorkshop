# from __future__ import absolute_import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as disp
from sklearn.neighbors import KNeighborsClassifier
import tqdm

MAX_TONNETZ_WEIGHT = 800
MAX_SPECTOGRAM_WEIGHT = 15000
MAX_TEMPOGRAM_WEIGHT = 10
MAX_CHROMA_WEIGHT = 10


class DatasetLoader:
    def __init__(self):
        self.X = []
        self.Y = []

    def load_dataset(self, path):
        for file in tqdm.tqdm(os.listdir(path), desc='Extracing audio features from data'):
            if os.path.isdir(path + '/' + file):
                for audio_file in os.listdir(path + '/' + file):
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
        spectogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
        tempogram = librosa.feature.tempogram(y=samples, sr=sample_rate)
        # spectogram = librosa.feature.melspectrogram(samples, sample_rate)
        return np.concatenate([tonnetz.flatten()[:MAX_TONNETZ_WEIGHT], spectogram.flatten()[:MAX_SPECTOGRAM_WEIGHT],
                               tempogram.flatten()[:MAX_TEMPOGRAM_WEIGHT], chroma_stft.flatten()[:MAX_CHROMA_WEIGHT]])


def test_accuracy(X, Y, classifier: KNeighborsClassifier):
    err = 0
    prediction = classifier.predict(X)
    for i in range(len(prediction)):
        y, pred = Y[i], prediction[i]
        print('model predicted:', pred, 'actual label:', y)
        if pred != y:
            err += 1
    return err / len(X)


def main():
    train_path = 'data/train'
    dataset_loader = DatasetLoader()
    dataset_loader.load_dataset(train_path)
    assert len(dataset_loader.X) == len(dataset_loader.Y)
    X, Y = shuffle(dataset_loader.X, dataset_loader.Y, random_state=42)
    knn_classifier = KNeighborsClassifier()

    knn_classifier.fit(X, Y)
    acc = test_accuracy(X, Y, knn_classifier)
    print('done with accuracy:', acc)


if __name__ == '__main__':
    main()
