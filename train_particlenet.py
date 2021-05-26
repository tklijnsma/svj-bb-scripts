import os
import os.path as osp
import math
import tqdm
import logging

import numpy as np
# import torch
# import torch.nn.functional as F

import bbefp
from time import strftime
from sklearn.metrics import confusion_matrix

import tensorflow.keras as keras

import tensorflow_addons as tfa


def main():
    N = 100 # Number of constituents per jet

    X_train, y_train = bbefp.dataset.get_data_particlenet('data/train/merged.npz', N=N)
    X_test, y_test = bbefp.dataset.get_data_particlenet('data/test/merged.npz', N=N)


    print('train:')
    print(X_train['points'].shape, X_train['features'].shape, X_train['mask'].shape, y_train.shape)
    print('test:')
    print(X_test['points'].shape, X_test['features'].shape, X_test['mask'].shape, y_test.shape)


    model = bbefp.particlenet.get_particle_net_lite(
        num_classes=2,
        input_shapes=dict(
            points=(N, 2),
            features=(N, 5),
            mask=(N,1)
            )
        )

    def lr_schedule(epoch):
        lr = 1e-4
        if epoch > 10:
            lr *= 0.1
        elif epoch > 20:
            lr *= 0.01
        logging.info('Learning rate: %f'%lr)
        return lr

    # Prepare callbacks for model saving and for learning rate adjustment.
    ckpt_dir = strftime('testckpts_particlenet_%b%d_%H%M%S')
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=ckpt_dir,
        monitor='val_accuracy', # Or 'val_acc', depending on versions (https://github.com/tensorflow/tensorflow/issues/33163)
        verbose=1,
        save_best_only=True
        )

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    # progress_bar = keras.callbacks.ProgbarLogger()
    callbacks = [
        checkpoint, lr_scheduler,
        # progress_bar
        ]

    model.compile(
        loss='categorical_crossentropy',
        # optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
        optimizer=tfa.optimizers.AdamW(weight_decay=.001, learning_rate=lr_schedule(0)),
        metrics=['accuracy']
        )
    model.summary()

    print(model)

    model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20, # --- train only for 1 epoch here for demonstration ---
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=callbacks
        )

if __name__ == '__main__':
    main()