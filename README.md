Music data analysis
===============
# Project description
The goal of this project is to determine the genre of a given music based on data related to it using data mining and machine learning tools.
From the GTZAN dataset, we use a CNN that trains with the image representation of music (spectrogram).
# How to use
## Requirements
* Python 3.7.x or Python 3.8.x
* TensorFlow
* Scikit-Learn
* Librosa
* Pydub
* Numpy
* Matplotlib
* Seaborn
* Pickle
* Pandas
* Optional: NVIDIA Cuda (for training on the GPU)
## Build
As the project already contains a trained CNN, the **Training** part is not necessary. You can go directly to the **General use (predictions)** part if you wish.
### Training
First, you need to download the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset at this address:  
http://marsyas.info/downloads/datasets.html (click on *Download the GTZAN genre collection*)  
  
Once the archive has been downloaded, run:  
```shell
tar -xvf [path_to_gtzan_archive]genres.tar.gz -C[path_to_project_root_directory]
cd [path_to_project_root_directory]
rm -rf *.mf
```
The third command allows you to delete useless files from the archive (that are not music).
The project tree should look like this:  
```
etc/
genres/
	blues/
	classical/
	[...]
	rock/
src/
.gitignore
README.md
```
  
For a CNN to be effective, it needs a large amount of data. To do this, you will split the music into several 4 second chunks with an overlap of 1 second. For this, just run this script in the root directory:  
`python src/split_music.py`
  
You now need to generate the dataset on which your CNN will train.  
Still in the root directory, run :  
`python src/data_generation.py`
(Be patient, this may take a while)  
*N.B. : The script will generate three files: two pickle files containing the spectrograms and labels for the CNN and a CSV file containing calculated variables (MFCC, RMSE, ...). The latter is not needed for the CNN, so you can skip this step by commenting out line 112 in **src/data_generation.py**: `save_calculated_variables_in_csv(music_directory)`*  
  
Now you will train and save the CNN and display the accuracy curve and the confusion matrix. For this, run in the root directory:  
`python src/CNN.py`  
(Be patient, this may take a while)  
  
### General use (predictions)
You can use the CNN in a concrete case. To do this, just run in the root directory:  
`python src/test_predictions.py [path_to_music_to_classify]`  
*N.B. 1: The music must be in .wav format. If this is not the case, you can use [Convertio.co](https://convertio.co/mp3-wav/) or any other tool of your choice to convert your music to the correct format.  
N.B. 2: Make sure that the file name does not contain any spaces. E.g.: my_music.wav*  
  
You will see the genre predicted by the CNN in the terminal.
# Team members
* Prof. Christophe ROSENBERGER, teacher and tutor
* Nathan MICHEL, project manager
* Raphaël ANCETTE, AI developer
* Camille GUIGNOL, AI developer
* Arnaud RIO, mobile developer
