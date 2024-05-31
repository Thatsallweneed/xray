import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from glob import glob

# Setting the GPU to be used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Path to the dataset
dataset_path = "D:\\Depaul\\DATA_SCIENCE\\prog_ml_apps\\DATASET\\archive (5)"
os.chdir(dataset_path)

# Load data
data = pd.read_csv('Data_Entry_2017.csv')
image_paths = {os.path.basename(x): x for x in glob(os.path.join('images*', 'images', '*.png'))}
data['path'] = data['Image Index'].map(image_paths.get)
data['Patient Age'] = data['Patient Age'].apply(lambda x: int(x[:-1]) if isinstance(x, str) else x)

# Preprocessing labels
data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
all_labels = np.unique(np.concatenate(data['Finding Labels'].map(lambda x: x.split('|')).values))
all_labels = [x for x in all_labels if len(x) > 0]
for label in all_labels:
    data[label] = data['Finding Labels'].apply(lambda x: 1.0 if label in x else 0)

# Selecting images with findings only
data = data[data['Finding Labels'] != '']

# Splitting data
train_df, valid_df = train_test_split(data, test_size=0.2, random_state=2021, stratify=data['Finding Labels'])

# Image Data Generator
IMG_SIZE = (128, 128)
idg = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, horizontal_flip=True)

def make_gen(df):
    return idg.flow_from_dataframe(
        dataframe=df,
        directory=None,  # already absolute paths in 'path'
        x_col='path',
        y_col='Finding Labels',
        class_mode='categorical',
        classes=list(all_labels),
        target_size=IMG_SIZE,
        batch_size=32,
        color_mode='rgb')

# Creating generators
train_gen = make_gen(train_df)
valid_gen = make_gen(valid_df)
test_X, test_Y = next(make_gen(valid_df))  # just as an example, should be separate test set

# DenseNet121 Model Function
def create_densenet121(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    for num_conv in [6, 12, 24, 16]:
        for _ in range(num_conv):
            x1 = layers.BatchNormalization()(x)
            x1 = layers.Activation('relu')(x1)
            x1 = layers.Conv2D(32, 3, padding='same')(x1)
            x = layers.Concatenate()([x, x1])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, 1, padding='same')(x)
        x = layers.AveragePooling2D(2, strides=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

# Setting up the model
with tf.device('/GPU:0'):
    model = create_densenet121(IMG_SIZE + (3,), len(all_labels))
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint("best_weights.h5", save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    callbacks_list = [checkpoint, early_stop]

    # Training
    model.fit(train_gen, epochs=10, validation_data=valid_gen, callbacks=callbacks_list)

# Predicting
pred_Y = model.predict(test_X, batch_size=32, verbose=1)

# ROC Curve
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (idx, label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), pred_Y[:, idx])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('training_results.png')
