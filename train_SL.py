import math
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import random

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.layers import Conv2D, Activation, Dense, Dropout, Softmax, LeakyReLU, BatchNormalization, AveragePooling2D, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
# from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError


def create_dir(dir_name):
    """Check if dir_name/ exits and create directory if not exits

    Args:
        dir_name (str): Name of dir to be created

    Returns:
        None: [description]
    """
    Path(dir_name).mkdir(parents=True, exist_ok=True)

    return None


def get_timestamp():
    """Create timestamp sting

    Returns:
        str: Timestamp
    """
    time_info = datetime.today()
    timestamp = f"{time_info.year}-{time_info.month}-{time_info.day}-{time_info.hour}-{time_info.minute}"

    return timestamp


def create_outputs_dir(tag=""):
    """Creates output direcotry template

    Args:
        tag (str, optional): Optional tag appended to output dir. Defaults to "".

    Returns:
        str: Path to output folder
    """
    # Get timestamp, append tag and create "outputs/output_dir_name"
    output_dir_name = f"outputs/{get_timestamp()}_{tag}"
    create_dir(output_dir_name)

    # Create empty subfolders in <timestamp>/ for valid_predictions/, real_predictions/
    create_dir(f"{output_dir_name}/valid_predictions")
    create_dir(f"{output_dir_name}/real_predictions")

    return output_dir_name


def draw_annot(image, x_0, y_0, gamma, conf, output_file="predicted.png"):
    plt.figure(figsize=[6, 6])
    plt.imshow(image)
    plt.scatter(x_0, y_0, color='r', s=40, marker="o")

    line_len = 20
    for x_1, y_1, angle, confi in zip(x_0, y_0, gamma, conf):
        x_2 = x_1 + line_len*math.cos(angle)
        y_2 = y_1 + line_len*math.sin(angle)
        plt.plot([x_1, x_2], [y_1, y_2], color='r')
    # plt.show()
    plt.savefig(output_file)


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
            y_train = np.zeros([batch_size, 16, 16, 1, 5])

            i = 0
            for batch_sample in batch_samples:
                img_name = batch_sample[0]
                ann_name = batch_sample[1]

                img = cv2.imread(str(img_name), 0)
                img = cv2.resize(img, (128, 128))
                img = img/255

                # storing x_values back to back
                x_train[i, :, :, 0] = img

                annot = np.loadtxt(ann_name)
                # finding which grid the object belongs to
                grid_x = annot[:, 1]//grid_w
                grid_y = annot[:, 2]//grid_h

                row = 0

                for cell_x, cell_y in zip(grid_x, grid_y):
                    if ((annot[row, 5] + annot[row, 6])*0.5) > 0.6:
                        y_train[i, int(cell_y), int(cell_x), 0, 0] = float(
                            annot[row, 1]/8 % 1)
                        y_train[i, int(cell_y), int(cell_x), 0,
                                1] = float(annot[row, 1]/8 % 1)
                        y_train[i, int(cell_y), int(cell_x), 0, 2] = float(
                            (math.sin(annot[row, 3]))**2)  # I_xx
                        y_train[i, int(cell_y), int(cell_x), 0, 3] = float(
                            (math.sin(2*annot[row, 4])+1)*0.5)  # I_xy
                        y_train[i, int(cell_y), int(cell_x),
                                0, 4] = 1.0  # conf
                        # y_train[i, int(cell_y), int(cell_x), 0, 5] = annot[row, 0]  # class
                        row = row+1

                i += 1

            yield x_train, y_train


def custom_loss(y_true, y_pred):

    DIRECTION_SCALE = 14.0
    COORD_SCALE = 4.0
    OBJECT_SCALE = 20.0
    NO_OBJECT_SCALE = 8.0
    # CLASS_SCALE = 1.0

    mask_shape = tf.shape(y_true)[: 4]

    coord_mask = tf.zeros(mask_shape)
    direction_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    # class_mask = tf.zeros(mask_shape)

    pred_xy = (y_pred[..., 0: 2])
    pred_exy = (y_pred[..., 2: 4])
    pred_conf = y_pred[..., 4]
    # pred_class = y_pred[..., 5:]

    true_xy = y_true[..., 0: 2]
    true_exy = y_true[..., 2: 4]
    true_conf = y_true[..., 4]
    # true_class = y_true[..., 5:]

    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    direction_mask = tf.expand_dims(y_true[..., 4], axis=-1) * DIRECTION_SCALE
    conf_mask = conf_mask + (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
    # class_mask = tf.expand_dims(y_true[..., 4], axis=-1) * CLASS_SCALE

    nb_coord_kpp = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32))
    nb_conf_kpp = tf.reduce_sum(tf.cast(conf_mask > 0.0, dtype=tf.float32))
    # nb_class_kpp = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype=tf.float32))

    loss_xy = tf.reduce_sum(tf.square(true_xy-pred_xy)
                            * coord_mask) / (nb_coord_kpp + 1e-6) / 2.
    loss_exy = tf.reduce_sum(tf.square(true_exy-pred_exy)
                             * direction_mask) / (nb_coord_kpp + 1e-6) / 2.
    loss_conf = tf.reduce_sum(
        tf.square(true_conf-pred_conf) * conf_mask) / (nb_conf_kpp + 1e-6) / 2.
    # loss_class = tf.reduce_sum(tf.square(true_class-pred_class) * class_mask) / (nb_class_kpp + 1e-6) / 2.

    # loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_class, logits=pred_class)
    # loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_coord_box + 1e-6)

    loss = loss_xy + loss_exy + loss_conf  # + loss_class

    return loss


def lr_scheduler(epoch, lr):
    decay_rate = 0.8
    decay_step = 30
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


def decode_annot(output):

    grid_w = 8
    grid_h = 8
    x_0 = []
    y_0 = []
    gamma = []
    conf = []
    # classes= []
    for i in range(16):
        for j in range(16):

            if output[i, j, 4] > 0.1:
                X = output[i, j, 0]*grid_w + grid_w*j
                x_0.append(X)

                Y = output[i, j, 1]*grid_h + grid_h*i
                y_0.append(Y)

                # angle = 0.5*math.asin((2*output[i, j, 3] - 1))
                angle = float(math.atan2(
                    output[i, j, 2] * 2, output[i, j, 3] * 2 - 1))
                gamma.append(angle)
                conf.append(output[i, j, 4])
                # classes.append(output[i,j,5])

    return x_0, y_0, gamma, conf  # , classes


def save_predictions(img_file_list, model, output_path, limit=10):
    """Predict and save images based on model

    Args:
        img_file_list (list): Image file paths
        model (tensorflow.model): Trained model
        output_path (str): Path to save images
        limit (int, optional): Maximum number of images to be predicted. Defaults to 10.

    """

    # Limiting the prediction dataset
    img_file_list = img_file_list[:limit]

    for img_f in img_file_list:
        img = cv2.imread(str(img_f), 0)
        img_ = cv2.resize(img, (128, 128))
        img_ = img_ / 255
        img_ = np.expand_dims(img_, axis=-1)
        img_ = np.expand_dims(img_, axis=0)
        out = model.predict(img_)
        reshaped = out.reshape((16, 16, 5))

        x, y, gamma, conf = decode_annot(reshaped)
        max_conf = np.max(reshaped[:, :, 4])
        print('max confidence = ', max_conf)
        draw_annot(img_, x, y, gamma, conf, f"{output_path}/{img_f.stem}.png")

    return None


def main():

    # Create output folder
    OUTPUT_DIR_PATH = create_outputs_dir()

    # Training params
    EPOCHS = 200
    BS = 8

    # Loading Data
    TRAIN_DATASET_PATH = Path("datasets/128_singlelayer/train")
    VALID_DATASET_PATH = Path("datasets/128_singlelayer/valid")
    REAL_DATASET_PATH = Path("datasets/real_shafts_sl")

    # TRAIN_DATASET_PATH = Path("datasets/train_10")
    # VALID_DATASET_PATH = Path("datasets/valid_10")
    # REAL_DATASET_PATH = Path("datasets/real_shafts_sl_10")

    train_img_paths = list(TRAIN_DATASET_PATH.glob("**/*.bmp"))
    train_ann_paths = list(TRAIN_DATASET_PATH.glob("**/*.txt"))
    real_img_paths = list(REAL_DATASET_PATH.glob("**/*.bmp"))

    valid_img_paths = list(VALID_DATASET_PATH.glob("**/*.bmp"))
    valid_ann_paths = list(VALID_DATASET_PATH.glob("**/*.txt"))

    num_train_samples = len(train_ann_paths)
    num_valid_samples = len(valid_img_paths)

    print('num of train samples: ', num_train_samples)
    print('num of valid samples: ', num_valid_samples)

    # Creating data generators
    train_datagen = data_generator(train_img_paths, train_ann_paths, BS)
    validation_datagen = data_generator(valid_img_paths, valid_ann_paths, BS)

    # Creating model
    # model = Sequential([
    #     Conv2D(16, (3, 3), padding='same', input_shape=(128, 128, 1)),
    #     BatchNormalization(),
    #     Activation('relu'),
    #     AveragePooling2D(2, 2),
    #     Conv2D(32, (3, 3), padding='same'),
    #     BatchNormalization(),
    #     Activation('relu'),
    #     AveragePooling2D(2, 2),
    #     Conv2D(5, (3, 3), padding='same'),
    #     BatchNormalization(),
    #     Activation('relu'),
    #     AveragePooling2D(2, 2),
    #     Reshape((16, 16, 1, 5))
    # ])

    model = tf.keras.Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(128, 128, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    for i in range(0, 20):
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(AveragePooling2D(2, 2))

    for i in range(0, 15):
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(AveragePooling2D(2, 2))

    for i in range(0, 15):
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(AveragePooling2D(2, 2))

    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Conv2D(5, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Softmax())

    # model.add(Conv2D(18, (3, 3), padding='same'))
    # model.add(Reshape(16, 16, 3, 6))
    model.add(Reshape((16, 16, 1, 5)))

    model.compile(optimizer=Adam(lr=0.0005),
                  loss=custom_loss,
                  metrics=['accuracy'])

    model.summary()

    # Fitting the model

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=300,  # 2or3
                               mode='min',
                               verbose=1)

    # checkpoint = ModelCheckpoint(saved_weights_name,
    #                                  monitor='val_loss',
    #                                  verbose=1,
    #                                  save_best_only=True,
    #                                  save_weights_only=False,
    #                                  mode='min',
    #                                  period=1)
    # tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
    #                               histogram_freq=0,
    #                               # write_batch_performance=True,
    #                               write_graph=True,
    #                               write_images=False)

    l_rate = LearningRateScheduler(lr_scheduler, verbose=1)

    history = model.fit(x=train_datagen,
                        validation_data=validation_datagen,
                        steps_per_epoch=num_train_samples,
                        validation_steps=num_valid_samples,
                        epochs=EPOCHS,
                        callbacks=[early_stop, l_rate],
                        verbose=1)

    # Saving the trained model
    model.save(f"{OUTPUT_DIR_PATH}/trained_SL.h5")

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
    plt.plot(epochs, acc, label='training')
    plt.plot(epochs, val_acc, label='validation')
    plt.legend(loc='upper left')
    plt.title('Training and validation accuracy')
    plt.savefig(f"{OUTPUT_DIR_PATH}/training-accurcy-plot.png")

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.figure()
    plt.plot(epochs, loss, label='training')
    plt.plot(epochs, val_loss, label='validation')
    plt.legend(loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig(f"{OUTPUT_DIR_PATH}/training-loss-plot.png")

    # ------------------------------------------------
    # PREDICTION
    # ------------------------------------------------
    save_predictions(valid_img_paths, model,
                     output_path=f"{OUTPUT_DIR_PATH}/valid_predictions")
    save_predictions(real_img_paths, model,
                     output_path=f"{OUTPUT_DIR_PATH}/real_predictions")

    return None


if __name__ == "__main__":

    main()
