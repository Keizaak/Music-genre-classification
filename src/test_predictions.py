# -*- coding: utf-8 -*-

from CNN import load_cnn_model
from split_music import split_music, export_slices, clear_create_directory
from dataset_generation import get_mel_spectrogram_from_music
import os
import sys


def get_occurrence_dict(list):
    dict_res = {}
    for e in list:
        if e not in dict_res.keys():
            dict_res[e] = 1
        else:
            dict_res[e] += 1
    return dict_res


def get_max_occurrence(list):
    dict = get_occurrence_dict(list)
    max_value = 0
    max_key = -1
    for key, value in dict.items():
        if max_value < value:
            max_value = value
            max_key = key

    return max_key


def split_predict_music(music_path):
    clear_create_directory("../chunks/")
    slices = split_music(music_path, duration=4000, overlapping=0)
    export_slices(slices, "../chunks/slices.wav")


def get_processed_music(music):
    spectrogram = get_mel_spectrogram_from_music(music)
    # Normalization
    spectrogram /= spectrogram.min()
    # Resize to : number of spectrogram * size of spectrogram * number of channels color (1 for grey)
    spectrogram = spectrogram.reshape(1, 128, 173, 1)
    return spectrogram


def predict_chunks_music(splices_path, model_cnn, categories):
    predicted_categories = []

    for file in os.listdir(splices_path):
        spectrogram = get_processed_music(splices_path + file)
        predictions = model_cnn.predict(spectrogram)[0].tolist()
        predicted_categories.append(predictions.index(max(predictions)))

    predicted = get_max_occurrence(predicted_categories)
    if predicted != -1:
        print(categories[predicted])
    else:
        print("Error: Could not predict")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: test_predictions.py path_to_music_wav")
        exit(1)

    music_path = sys.argv[1]

    if music_path[-4:] != ".wav":
        print("Error: the music must in .wav format")
        exit(1)

    labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    model = load_cnn_model()
    split_predict_music(music_path)
    predict_chunks_music("../chunks/", model, labels)
