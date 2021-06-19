import math
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import random

from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization, AveragePooling2D, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError


def filter_annot(annot_path, check_params=[128]):
    annot = np.loadtxt(annot_path)

    # checking for number of data points
    if annot.shape[1] != 7:
        return False

    # checking for x limits
    if (annot[:, 1].max() > 128) or (annot[:, 1].min() < 0):
        return False

    # checking for y limits
    if (annot[:, 2].max() > 128) or (annot[:, 2].min() < 0):
        return False

    return True


def shuffle_sample(img_paths, ann_paths):
    samples = []
    # storing image_paths and corresponding annotation paths as a pair
    for I_Path, A_Path in zip(img_paths, ann_paths):
        if filter_annot(A_Path):
            samples.append([I_Path, A_Path])
    # shuffle the sample
    random.shuffle(samples)
    return samples


def store_annot(i, cell_x, cell_y, index, annot, row, y_train):
    grid_w = 8
    grid_h = 8
    y_train[i, int(cell_x), int(cell_y), index, 0] = (
        annot[row, 1] - grid_w * int(cell_x)) / grid_w  # x_0
    y_train[i, int(cell_x), int(cell_y), index, 1] = (
        annot[row, 2] - grid_h * int(cell_y)) / grid_h  # y_0
    y_train[i, int(cell_x), int(cell_y), index, 2] = (
        math.sin(annot[row, 3]))**2  # I_xx
    y_train[i, int(cell_x), int(cell_y), index, 3] = (
        math.sin(2*annot[row, 4])+1)*0.5  # I_xy
    y_train[i, int(cell_x), int(cell_y), index, 4] = (
        annot[row, 5] + annot[row, 6]) * 0.5  # conf
    y_train[i, int(cell_x), int(cell_y), index, 5] = annot[row, 0]  # class

    return y_train


def data_generator(img_paths, ann_paths, batch_size):
    grid_h = 8
    grid_w = 8

    # get the shuffled samples
    samples = shuffle_sample(img_paths, ann_paths)
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]

            # Initialise X_train and y_train arrays for this batch
            x_train = np.zeros([batch_size, 128, 128, 1])
            y_train = np.zeros([batch_size, 16, 16, 3, 6])

            i = 0
            for batch_sample in batch_samples:
                img_name = batch_sample[0]
                ann_name = batch_sample[1]

                img = cv2.imread(str(img_name), 0)
                img = cv2.resize(img, (128, 128))
                img = img/128

                # storing x_values back to back
                x_train[i, :, :, 0] = img

                annot = np.loadtxt(ann_name)
                # finding which grid the object belongs to
                grid_x = annot[:, 1]//grid_w
                grid_y = annot[:, 2]//grid_h

                row = 0

                for cell_x, cell_y in zip(grid_x, grid_y):
                    if y_train[i, int(cell_x), int(cell_y), 0, 4] == 0:
                        y_train = store_annot(i, cell_x, cell_y, 0,
                                              annot, row, y_train)

                    elif y_train[i, int(cell_x), int(cell_y), 1, 4] == 0:
                        y_train = store_annot(i, cell_x, cell_y, 1,
                                              annot, row, y_train)

                    else:
                        y_train = store_annot(i, cell_x, cell_y, 2,
                                              annot, row, y_train)

                    row = row+1

                i += 1

            yield x_train, y_train


def main():

    # Training params

    EPOCHS = 10
    BS = 2

    # Loading Data
    TRAIN_DATASET_PATH = Path("datasets/128_multilayer/train")
    VALID_DATASET_PATH = Path("datasets/128_multilayer/valid")
    # TRAIN_DATASET_PATH = Path("datasets/practice/train_m")
    # VALID_DATASET_PATH = Path("datasets/practice/valid_m")

    train_img_paths = list(TRAIN_DATASET_PATH.glob("**/*.bmp"))
    train_ann_paths = list(TRAIN_DATASET_PATH.glob("**/*.txt"))

    valid_img_paths = list(VALID_DATASET_PATH.glob("**/*.bmp"))
    valid_ann_paths = list(VALID_DATASET_PATH.glob("**/*.txt"))

    num_train_samples = len(train_ann_paths)
    num_valid_samples = len(valid_img_paths)

    print('num of train samples: ', num_train_samples)
    print('num of valid samples: ', num_valid_samples)

    # Creating data generators
    train_datagen = data_generator(train_img_paths, train_ann_paths, BS)
    validation_datagen = data_generator(valid_img_paths, valid_ann_paths, BS)

    # Create model
    model = Sequential([
        Conv2D(16, (3, 3), padding='same', input_shape=(128, 128, 1)),
        BatchNormalization(),
        Activation('relu'),
        AveragePooling2D(2, 2),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        AveragePooling2D(2, 2),
        Conv2D(18, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        AveragePooling2D(2, 2),
        Reshape((16, 16, 3, 6))
    ])

    model.compile(optimizer=Adam(lr=0.001),
                  loss=MeanSquaredError(),
                  metrics=['accuracy'])

    model.summary()

    # Fitting the model

    history = model.fit(x=train_datagen,
                        validation_data=validation_datagen,
                        steps_per_epoch=num_train_samples,
                        validation_steps=num_valid_samples,
                        epochs=EPOCHS,
                        verbose=1)

    # Saving the trained model
    model.save("trained.h5")

    # Evaluating Accuracy and Loss

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.figure()
    plt.plot(epochs,     acc,   label='training')
    plt.plot(epochs, val_acc, label='validation')
    plt.legend(loc='upper left')
    plt.title('Training and validation accuracy')
    plt.savefig("training-accurcy-plot.png")

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.figure()
    plt.plot(epochs, loss,    label='training')
    plt.plot(epochs, val_loss, label='validation')
    plt.legend(loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig("training-loss-plot.png")

    return None


if __name__ == "__main__":

    main()
