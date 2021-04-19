# -*- coding: utf-8 -*-

import os
import librosa
import pickle
import numpy
import pandas


def get_mel_spectrogram_from_music(audio_file):
    y, sr = librosa.load(audio_file)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S = librosa.power_to_db(S, ref=numpy.max)
    return S


def get_calculated_variables_from_music(audio_file):
    y, sr = librosa.load(audio_file)

    res = []
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # n_mfcc * t numpy matrix
    for mfcc_coeff in mfcc:
        mfcc_coeff = mfcc_coeff.reshape(1, mfcc_coeff.shape[0]).copy()
        res.append(mfcc_coeff)

    # RMSE (Root Mean Square Energy)
    rmse = librosa.feature.rms(y=y)  # a 1 * t numpy matrix
    res.append(rmse)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y)  # a 1 * t numpy matrix
    # zcr[0, i] is the fraction of zero crossings in the ith frame
    res.append(zcr)

    # Centroids
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)  # a 1 * t numpy matrix
    res.append(centroids)

    # Spectral flatness
    s_flat = librosa.feature.spectral_flatness(y=y)  # a 1 * t numpy matrix
    res.append(s_flat)

    return res


def save_spectrograms_in_pkl(audio_directory):
    x_file = open("../dataset/x_mel_spectrogram.pkl", "wb")
    y_file = open("../dataset/y_labels.pkl", "wb")

    x = []
    y = []
    label_dict = {
        "blues": 0,
        "classical": 1,
        "country": 2,
        "disco": 3,
        "hiphop": 4,
        "jazz": 5,
        "metal": 6,
        "pop": 7,
        "reggae": 8,
        "rock": 9
    }

    for directory in os.listdir(audio_directory):
        if os.path.isdir(os.path.join(audio_directory, directory)) and directory[0] != '.':
            for file in os.listdir(audio_directory + "/" + directory):
                file_path = audio_directory + "/" + directory + "/" + file

                spectrogram = get_mel_spectrogram_from_music(file_path)
                x.append(spectrogram)

                y.append(directory)

    x = numpy.array(x)
    y = pandas.Series(y)
    y = y.map(label_dict).values

    pickle.dump(x, x_file, pickle.HIGHEST_PROTOCOL)
    pickle.dump(y, y_file, pickle.HIGHEST_PROTOCOL)
    x_file.close()
    y_file.close()


def load_data_from_file(file_name):
    file = open(file_name, "rb")
    data = pickle.load(file)
    file.close()
    return data


def save_calculated_variables_in_csv(audio_directory):
    x = pandas.DataFrame()
    # x = []
    for directory in os.listdir(audio_directory):
        if os.path.isdir(os.path.join(audio_directory, directory)) and directory[0] != '.':
            for file in os.listdir(audio_directory + "/" + directory):
                file_path = audio_directory + "/" + directory + "/" + file
                var = get_calculated_variables_from_music(file_path)
                var = numpy.array(var).flatten()
                df = pandas.DataFrame(var).transpose()
                df["genre"] = directory
                x = x.append(df)
                print(file)
    x.to_csv("../dataset/data_calculated_variables.csv", index=None, sep=';')


if __name__ == "__main__":
    music_directory = "../splits/"
    save_spectrograms_in_pkl(music_directory)
    save_calculated_variables_in_csv(music_directory)
