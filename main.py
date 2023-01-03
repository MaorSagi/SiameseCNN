import numpy as np
import pandas as pd
import os
from keras import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l2
from keras.models import load_model
from keras.optimizers import Adam
import json
import time
import matplotlib.pyplot as plt
from tensorflow import keras

DATA_PATH = "lfwa/lfw2/lfw2"

BATCHNORM_AFTER_RELU = 0
BATCHNORM_BEFORE_RELU = 1
BATCHNORM_BETWEEN_ALL = 2
RUNNING_EXECUTION_NAME = '1. First try, strides=2'


class MyModelCheckpoint(ModelCheckpoint):

    def _init_(self, *args, **kwargs):
        super(MyModelCheckpoint, self)._init_(*args, **kwargs)

    # redefine the save so it only activates after 100 epochs
    def on_epoch_end(self, epoch, logs=None):
        if epoch > 70: super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)


def load_data(path, sizes):
    """
    load data from a given path
    :param path: path to dataset
    :param sizes: image sizes
    :return: dictionary of the images by classes
    """

    image_dict = {}
    count = 0
    for dir in os.listdir(path):
        # load an image from file
        for file in os.listdir(path + '/' + dir):
            count += 1
            file_path = path + '/' + dir + '/' + file
            image = load_img(file_path, target_size=(sizes, sizes), grayscale=True)
            # to array
            image = img_to_array(image)
            image = image / 255.

            if dir not in image_dict.keys():
                image_dict[dir] = {}
            # key for each person in the data
            image_dict[dir][int(file[len(dir) + 1:-4])] = image
    return image_dict


def create_train_test_split(image_dict):
    train_match = pd.read_csv('./pairsDevTrain.txt', sep='\t', header=None,
                              skiprows=[0] + list(range(1101, 2240)))
    test_match = pd.read_csv('./pairsDevTest.txt', sep='\t', header=None,
                             skiprows=[0] + list(range(501, 1101)))
    train_mismatch = pd.read_csv('./pairsDevTrain.txt', sep='\t', header=None,
                                 skiprows=[0] + list(range(0, 1101)))
    test_mismatch = pd.read_csv('./pairsDevTest.txt', sep='\t', header=None,
                                skiprows=[0] + list(range(0, 501)))

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for _, (name, i, j) in train_match.iterrows():
        x_train.append([image_dict[name][i], image_dict[name][j]])
        y_train.append(1)

    for _, (name, i, j) in test_match.iterrows():
        x_test.append([image_dict[name][i], image_dict[name][j]])
        y_test.append(1)

    for _, (name1, i, name2, j) in train_mismatch.iterrows():
        x_train.append([image_dict[name1][i], image_dict[name2][j]])
        y_train.append(0)

    for _, (name1, i, name2, j) in test_mismatch.iterrows():
        x_test.append([image_dict[name1][i], image_dict[name2][j]])
        y_test.append(0)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def validation_set_split(val_size, train_size, train_a_x, train_b_x, y_train):
    train_a_x_match = train_a_x[:train_size]
    train_b_x_match = train_b_x[:train_size]
    y_train_match = y_train[:train_size]

    train_a_x_mismatch = train_a_x[train_size:]
    train_b_x_mismatch = train_b_x[train_size:]
    y_train_mismatch = y_train[train_size:]

    val_a_x_match = train_a_x_match[:val_size]
    val_a_x_mismatch = train_a_x_mismatch[:val_size]

    val_b_x_match = train_b_x_match[:val_size]
    val_b_x_mismatch = train_b_x_mismatch[:val_size]

    y_val_match = y_train_match[:val_size]
    y_val_mismatch = y_train_mismatch[:val_size]

    train_a_x_match = train_a_x_match[val_size:]
    train_a_x_mismatch = train_a_x_mismatch[val_size:]

    train_b_x_match = train_b_x_match[val_size:]
    train_b_x_mismatch = train_b_x_mismatch[val_size:]

    y_train_match = y_train_match[val_size:]
    y_train_mismatch = y_train_mismatch[val_size:]

    val_a_x = np.concatenate([val_a_x_match, val_a_x_mismatch])
    val_b_x = np.concatenate([val_b_x_match, val_b_x_mismatch])
    y_val = np.concatenate([y_val_match, y_val_mismatch])

    train_a_x = np.concatenate([train_a_x_match, train_a_x_mismatch])
    train_b_x = np.concatenate([train_b_x_match, train_b_x_mismatch])
    y_train = np.concatenate([y_train_match, y_train_mismatch])

    return [train_a_x, train_b_x], y_train, [val_a_x, val_b_x], y_val


def add_data_augmentation(train_a_x, train_b_x, y_train):
    datagen = ImageDataGenerator(zoom_range=[0.88, 1.02], brightness_range=[0.95, 1.05],
                                 horizontal_flip=False)
    train_a_x_aug1 = [(datagen.flow(aug.reshape([1, 105, 105, 1])))[0].squeeze(0) for aug in train_a_x]
    train_b_x_aug1 = [(datagen.flow(aug.reshape([1, 105, 105, 1])))[0].squeeze(0) for aug in train_b_x]
    train_a_x_aug2 = [(datagen.flow(aug.reshape([1, 105, 105, 1])))[0].squeeze(0) for aug in train_a_x]
    train_b_x_aug2 = [(datagen.flow(aug.reshape([1, 105, 105, 1])))[0].squeeze(0) for aug in train_b_x]
    combined_train_a_x = np.concatenate([np.array(train_a_x_aug1) / 255, np.array(train_a_x_aug2) / 255, train_a_x])
    combined_train_b_x = np.concatenate([np.array(train_b_x_aug1) / 255, np.array(train_b_x_aug2) / 255, train_b_x])
    combined_y_train = np.concatenate([y_train, y_train, y_train])
    return combined_train_a_x, combined_train_b_x, combined_y_train


def add_lable_clipping(y_train, exec_on_train):
    # Label clipping
    k = 2
    eps = 0.03
    y_train_clipped1 = []
    y_train_clipped2 = []

    for i in y_train:
        if i == 0:
            y_train_clipped1.append(eps / k)
        else:
            y_train_clipped1.append(1 - ((k - 1) / k) * eps)
    if exec_on_train:
        eps = 0.01
        for i in y_train:
            if i == 0:
                y_train_clipped2.append(eps / k)
            else:
                y_train_clipped2.append(1 - ((k - 1) / k) * eps)
        return np.concatenate([y_train_clipped1, y_train_clipped1, y_train_clipped2])
    return np.concatenate([y_train_clipped1, y_train_clipped1, y_train])


def scheduler(epoch, lr):
    if epoch > 40:
        return 0.0001
    else:
        return lr


def L1_siamese(h1, h2):
    return keras.backend.abs(h1 - h2)


def create_full_network(batchnorm=None):
    model = Sequential()
    model.add(Conv2D(64, (10, 10),
                     kernel_initializer=RandomNormal(mean=0.0, stddev=(2 / ((10 ** 2) * 64)) ** 0.5),
                     bias_initializer=RandomNormal(mean=0, stddev=0.01),
                     kernel_regularizer=l2(0.01)))
    if batchnorm and batchnorm == 1:
        model.add(BatchNormalization())
    model.add(ReLU())
    if batchnorm and batchnorm % 2 == 0:
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2))
    if batchnorm and batchnorm % 2 == 0:
        model.add(BatchNormalization())
    model.add(Conv2D(128, (7, 7),
                     kernel_initializer=RandomNormal(mean=0.0, stddev=(2 / ((7 ** 2) * 128)) ** 0.5),
                     bias_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=l2(0.01)))
    if batchnorm and batchnorm == 1:
        model.add(BatchNormalization())
    model.add(ReLU())
    if batchnorm and batchnorm % 2 == 0:
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2))
    if batchnorm and batchnorm % 2 == 0:
        model.add(BatchNormalization())
    model.add(Conv2D(128, (4, 4),
                     kernel_initializer=RandomNormal(mean=0.0, stddev=(2 / ((4 ** 2) * 128)) ** 0.5),
                     bias_initializer=RandomNormal(mean=0, stddev=0.01),
                     kernel_regularizer=l2(0.01)))
    if batchnorm and batchnorm == 1:
        model.add(BatchNormalization())
    model.add(ReLU())
    if batchnorm and batchnorm % 2 == 0:
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2))
    if batchnorm and batchnorm % 2 == 0:
        model.add(BatchNormalization())
    model.add(Conv2D(256, (4, 4),
                     kernel_initializer=RandomNormal(mean=0.0, stddev=(2 / ((4 ** 2) * 256)) ** 0.5),
                     bias_initializer=RandomNormal(mean=0, stddev=0.01),
                     kernel_regularizer=l2(0.01)))
    if batchnorm and batchnorm == 1:
        model.add(BatchNormalization())
    model.add(ReLU())
    if batchnorm and batchnorm % 2 == 0:
        model.add(BatchNormalization())  # model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=(2 / 4096) ** 0.5),
                    bias_initializer=RandomNormal(mean=0, stddev=0.01),
                    kernel_regularizer=l2(0.01)))

    image_1 = Input((105, 105, 1))
    image_2 = Input((105, 105, 1))

    embd_image_1 = model(image_1)
    embd_image_2 = model(image_2)

    distace = L1_siamese(embd_image_1, embd_image_2)
    last_layer = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=(2 / 1) ** 0.5),
                       bias_initializer=RandomNormal(mean=0, stddev=0.01),
                       kernel_regularizer=l2(0.01))(distace)

    full_network = Model(inputs=[image_1, image_2], outputs=last_layer)
    return full_network


def save_model_results_on_json_file(full_network, file_name):
    with open(file_name + ".json", "a") as file:
        json.dump(full_network.history.history, file)
    json_obj = full_network.history.history
    for key in json_obj:
        if isinstance(json_obj[key], list):
            for i in range(len(json_obj[key])):
                json_obj[key][i] = str(json_obj[key][i])
        else:
            json_obj[key] = str(json_obj[key])


def load_model_results_from_json(file_name):
    with open(file_name + ".json") as file:
        json_obj = json.load(file)
        for key in json_obj:
            if isinstance(json_obj[key], list):
                for i in range(len(json_obj[key])):
                    json_obj[key][i] = float(json_obj[key][i])
            else:
                json_obj[key] = float(json_obj[key])
        return json_obj


def plot_graphs(json_obj, best_full_network=None, best_model_checkpoint=None):
    epochs_es = len(json_obj['loss'])
    plt.plot(np.arange(epochs_es), json_obj['loss'], label='Training')
    plt.plot(np.arange(epochs_es), json_obj['val_loss'], label='Validation')
    if best_full_network:
        plt.axvline(x=best_model_checkpoint, label='Best model checkpoint', c='r')

    plt.xlabel('Ephoc')
    plt.ylabel('Loss')
    plt.title(RUNNING_EXECUTION_NAME + ' Results')
    plt.legend()
    plt.show()

    plt.plot(np.arange(epochs_es), json_obj['binary_accuracy'], label='Training')
    plt.plot(np.arange(epochs_es), json_obj['val_binary_accuracy'], label='Validation')
    if best_full_network:
        plt.axvline(x=best_model_checkpoint, label='Best model checkpoint', c='r')

    plt.xlabel('Ephoc')
    plt.ylabel('Accuracy')
    plt.title(RUNNING_EXECUTION_NAME + ' Results')
    plt.legend()
    plt.show()


def evaluate(x_test, y_test, best_full_network=None, best_file_model_name=None):
    if best_full_network:
        best_full_network = load_model(best_file_model_name)
        return best_full_network.evaluate(x_test, y_test)
    full_network = load_model("./models/full_network.h5")
    return full_network.evaluate(x_test, y_test)


def preprocessing_and_training_model(x_train, y_train, epochs, optimizer, batchnorm, callbacks=None,
                                     augmentation=None, lable_clipping=None, exec_lable_clipping_on_train=None):
    train_set_size = int(len(x_train) / 2)  # contain match and mismatch pairs

    # Input images pairs separation
    train_a_x = x_train[:, 0, :, :, :]
    train_b_x = x_train[:, 1, :, :, :]

    # Data prepared to 10-cross validation, validation set size has to be a product of 110=train_set_size*0.1
    val_set_size = int(2 * (0.1 * train_set_size))
    [train_a_x, train_b_x], y_train, [val_a_x, val_b_x], y_val = validation_set_split(val_set_size, train_set_size,
                                                                                      train_a_x, train_b_x, y_train)

    if augmentation and lable_clipping:
        y_train_lable_clipping = add_lable_clipping(y_train, exec_lable_clipping_on_train)

    if augmentation:
        train_a_x, train_b_x, y_train = add_data_augmentation(train_a_x, train_b_x, y_train)

    if lable_clipping:
        y_train = y_train_lable_clipping

    full_network = create_full_network(batchnorm)
    full_network.compile(loss='binary_crossentropy', metrics=['binary_accuracy']
                         , optimizer=optimizer)
    start_time = time.time()
    full_network.fit([train_a_x, train_b_x], y_train, epochs=epochs, batch_size=64,
                     validation_data=([val_a_x, val_b_x], y_val), callbacks=callbacks)
    end_time = time.time()
    training_time = end_time - start_time
    epochs_es = len(full_network.history.history['loss'])
    full_network.history.history['training_time'] = training_time
    full_network.history.history['training_time_per_epoch'] = training_time / epochs_es
    return full_network


if __name__ == '__main__':
    augmentation = None
    lable_clipping = None
    exec_lable_clipping_on_train = None
    training_exec = False
    evaluate_exec = True
    model_results_file_path = "./jsons/8. Augmentation,patience=30 with batchnorm only after ReLU, clipping"
    best_model_h5_path = None#'./models/best_model_epoch_80.h5'
    best_model_checkpoint = None#80
    best_full_network = False

    # Delete  best model from last executions
    filelist = [f for f in os.listdir('./models')]
    for f in filelist:
        os.remove(os.path.join('./models', f))

    images = load_data(DATA_PATH, 105)
    x_train, y_train, x_test, y_test = create_train_test_split(images)
    test_a_x = x_test[:, 0, :, :, :]
    test_b_x = x_test[:, 1, :, :, :]

    if training_exec:
        epochs = 200
        period = 5
        patience = 5
        batchnorm = None
        es = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
        mc = MyModelCheckpoint('./models/best_model_epoch_{epoch}.h5', period=period,
                               monitor='val_loss', mode='min', save_best_only=True)
        scd = LearningRateScheduler(scheduler)
        callbacks = []
        optimizer = Adam(0.001)

        full_network = preprocessing_and_training_model(x_train, y_train, epochs, optimizer, batchnorm, callbacks,
                                                        augmentation, lable_clipping, exec_lable_clipping_on_train)
        full_network.save("./models/full_network.h5")
        save_model_results_on_json_file(full_network, './jsons/'+RUNNING_EXECUTION_NAME)

    if evaluate_exec:
        model_json = load_model_results_from_json(model_results_file_path)
        plot_graphs(model_json, best_full_network, best_model_checkpoint)
        evaluate([test_a_x, test_b_x], y_test, best_full_network, best_model_h5_path)
