# -*- coding: utf-8 -*-

import pydub
import os
import shutil


def split_music(music_path, duration, overlapping):
    music = pydub.AudioSegment.from_wav(music_path)
    slices = []

    for i in range(0, len(music) - overlapping, duration - overlapping):
        slices.append(music[i: i + duration])
    slices.pop(-1)
    return slices


def export_slices(slices, path):
    file_name = path[:-4]  # We remove the last 4 characters ".wav" to get the file name
    for i, slice in enumerate(slices):
        slice.export(file_name + "." + str(i) + ".wav", format="wav")


def clear_create_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)


def split_all_musics(music_directory, split_directory, duration, overlapping):
    clear_create_directory(split_directory)
    for directory in os.listdir(music_directory):
        os.mkdir(split_directory + directory)
        if os.path.isdir(os.path.join(music_directory, directory)) and directory[0] != '.':
            for file in os.listdir(music_directory + directory):
                file_path = directory + "/" + file
                slices = split_music(music_directory + file_path, duration, overlapping)
                export_slices(slices, split_directory + file_path)


if __name__ == "__main__":
    music_directory = "../genres/"
    split_directory = "../splits/"
    split_all_musics(music_directory, split_directory, duration=4000, overlapping=1000)
