import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# Set seaborn style
sns.set(rc={'figure.figsize': (10, 7)}, style='darkgrid')
sns.set_color_codes()

# Ignore warnings
warnings.filterwarnings('ignore')

def load_data(data_path, images_path):
    os.chdir(data_path)
    data = pd.read_csv('Data_Entry_2017.csv')
    my_glob = glob.glob(images_path)
    all_image_paths = {os.path.basename(x): x for x in my_glob}
    
    print('Scans found:', len(all_image_paths), ', Total Headers', data.shape[0])
    data['path'] = data['Image Index'].map(all_image_paths.get)
    data['Patient Age'] = data['Patient Age'].map(lambda x: int(x[:-1]) if isinstance(x, str) else x)
    return data

def process_labels(data):
    data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]
    
    for c_label in all_labels:
        if len(c_label) > 1:
            data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
    
    MIN_CASES = 1000
    all_labels = [c_label for c_label in all_labels if data[c_label].sum() > MIN_CASES]
    
    sample_weights = data['Finding Labels'].map(lambda x: len(x.split('|')) if len(x) > 0 else 0).values + 4e-2
    sample_weights /= sample_weights.sum()
    data = data.sample(100000, weights=sample_weights)
    
    data = data.drop(['Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position',
                      'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11'], axis=1)
    
    data['disease_vec'] = data.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
    
    counts = data['Finding Labels'].value_counts()
    mask = data['Finding Labels'].isin(counts[counts >= 251].index)
    data = data[mask]
    
    return data, all_labels

def split_data(data):
    train_df, valid_df = train_test_split(data,
                                          test_size=0.25,
                                          random_state=2018,
                                          stratify=data['Finding Labels'].map(lambda x: x[:4]))
    return train_df, valid_df

def create_image_generators(train_df, valid_df, all_labels, img_size):
    core_idg = ImageDataGenerator(samplewise_center=True,
                                  samplewise_std_normalization=True,
                                  horizontal_flip=True,
                                  vertical_flip=False,
                                  height_shift_range=0.05,
                                  width_shift_range=0.1,
                                  rotation_range=5,
                                  shear_range=0.1,
                                  fill_mode='reflect',
                                  zoom_range=0.15)
    
    train_df['newLabel'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
    valid_df['newLabel'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
    
    train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='newLabel',
                                             class_mode='categorical',
                                             classes=all_labels,
                                             target_size=img_size,
                                             color_mode='rgb',
                                             batch_size=32)
    
    valid_gen = core_idg.flow_from_dataframe(dataframe=valid_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='newLabel',
                                             class_mode='categorical',
                                             classes=all_labels,
                                             target_size=img_size,
                                             color_mode='rgb',
                                             batch_size=256)
    
    return train_gen, valid_gen

def dense_block(x, blocks):
    for _ in range(blocks):
        x1 = layers.BatchNormalization()(x)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Conv2D(32, kernel_size=3, padding='same')(x1)
        x = layers.Concatenate()([x, x1])
    return x

def transition_block(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, kernel_size=1, padding='same')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    return x

def create_densenet121(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = dense_block(x, 6)
    x = transition_block(x)

    x = dense_block(x, 12)
    x = transition_block(x)

    x = dense_block(x, 24)
    x = transition_block(x)

    x = dense_block(x, 16)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model):
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'mae'])
    return model

def train_model(model, train_gen, valid_gen):
    weight_path = "xray_class_weights.best.weights.h5"
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=5)
    callbacks_list = [checkpoint, early]
    
    model.fit(train_gen, validation_data=valid_gen, epochs=50, callbacks=callbacks_list)

def plot_roc_curve(model, valid_gen, all_labels):
    test_X, test_Y = next(valid_gen)
    pred_Y = model.predict(test_X, batch_size=32, verbose=True)
    
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    for (idx, c_label) in enumerate(all_labels):
        fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), pred_Y[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    fig.savefig('barely_trained_net.png')

def main():
    data_path = "D:/Depaul/DATA_SCIENCE/prog_ml_apps/DATASET/archive (5)"
    images_path = 'images*/images/*.png'
    
    data = load_data(data_path, images_path)
    data, all_labels = process_labels(data)
    train_df, valid_df = split_data(data)
    img_size = (128, 128)
    train_gen, valid_gen = create_image_generators(train_df, valid_df, all_labels, img_size)
    
    input_shape = (128, 128, 3)
    num_classes = len(all_labels)
    model = create_densenet121(input_shape, num_classes)
    model = compile_model(model)
    
    train_model(model, train_gen, valid_gen)
    plot_roc_curve(model, valid_gen, all_labels)

if __name__ == "__main__":
    main()
