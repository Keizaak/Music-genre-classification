# -*- coding: utf-8 -*-

import dataset_generation
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn


def create_model():
    # Initializing an empty neural network
    modelCNN = models.Sequential()

    # kernel_size: size of the pixel bloc treated (here 3*3)
    # filters: depth of the layer
    modelCNN.add(layers.Conv2D(filters=32,
                               kernel_size=(3, 3),
                               activation="relu",
                               input_shape=(128, 173, 1)))
    # MaxPooling: taking the maximum value of the pixels in a bloc of 2*4
    modelCNN.add(layers.MaxPooling2D(pool_size=(2, 4)))

    modelCNN.add(layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation="relu"))
    modelCNN.add(layers.MaxPooling2D(pool_size=(2, 4)))

    modelCNN.add(layers.Conv2D(filters=128,
                               kernel_size=(3, 3),
                               activation="relu"))
    modelCNN.add(layers.MaxPooling2D(pool_size=(2, 4)))

    # Flatten layer: transform the matrix of parameters into a vector that can be understood by a neural network
    modelCNN.add(layers.Flatten())

    # Dense layer: classic layer of fully connected neurons (here 128)
    modelCNN.add(layers.Dense(128, activation="relu"))
    # Dropout layer: deliberately ignore neurons with a probability of 50%
    # Avoid local optimizations and reduce overfitting
    modelCNN.add(layers.Dropout(0.5))
    # Final layer: classification layer
    # Softmax = each neuron has probability that the music belongs to this genre. Take the highest probability
    modelCNN.add(layers.Dense(10, activation="softmax"))
    modelCNN.summary()

    # Compiling our neural network
    modelCNN.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=tf.keras.metrics.SparseCategoricalAccuracy())

    return modelCNN


def train_model(model_cnn):
    X = dataset_generation.load_data_from_file("../dataset/x_mel_spectrogram.pkl")
    y = dataset_generation.load_data_from_file("../dataset/y_labels.pkl")

    # Split the musics. 80% for learning and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.20)
    # Scaling our data to be between 0 and 1 using the minimum value returned by X_train.min()
    X_train /= -80
    X_test /= -80
    # Reshaping spectrogram to be 128 * 173 * 1, where 1 represents the single color channel
    print(X_train.shape[0])
    X_train = X_train.reshape(X_train.shape[0], 128, 173, 1)
    X_test = X_test.reshape(X_test.shape[0], 128, 173, 1)

    # Fitting our neural network
    # Execution time: around 10s/epoch (on gpu)
    # batch_size = number of musics used per epoch
    history = model_cnn.fit(X_train,
                            y_train,
                            validation_data=(X_test, y_test),
                            epochs=20,
                            batch_size=16,
                            verbose=1)

    test_loss, test_acc = model_cnn.evaluate(X_test, y_test, verbose=1)
    print("Accuracy score: " + str(test_acc))

    show_accuracy_evolution(history)
    show_confusion_matrix(model_cnn, X_test, y_test)

    save_cnn_model(model_cnn)


def show_accuracy_evolution(history):
    train_loss = history.history["sparse_categorical_accuracy"]
    test_loss = history.history["val_sparse_categorical_accuracy"]

    plt.figure(figsize=(12, 8))
    plt.plot(train_loss, label="Training", color="red")
    plt.plot(test_loss, label="Testing", color="blue")
    plt.title("Training and testing accuracy by epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def show_confusion_matrix(model, X_test, y_test):
    labels_dict = {
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

    # Making predictions from the model
    predictions = model.predict(X_test, verbose=1)
    # Calculating the confusion matrix
    confusion_matrix = tf.math.confusion_matrix(labels=y_test,
                                                predictions=tf.argmax(predictions, 1))

    plt.figure(figsize=(8, 8))
    ax = seaborn.heatmap(confusion_matrix, square=True, annot=True, cmap=seaborn.cubehelix_palette(50),
                         xticklabels=labels_dict.keys(),
                         yticklabels=labels_dict.keys())
    plt.xlabel("Predicted genres")
    plt.ylabel("Real genres")
    plt.show()


def save_cnn_model(model):
    model.save("cnn_model")


def load_cnn_model():
    return models.load_model("cnn_model")


if __name__ == "__main__":
    model = create_model()
    train_model(model)
