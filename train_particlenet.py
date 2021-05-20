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


def main():
    X_train, y_train = bbefp.dataset.get_data_particlenet('data/train/merged.npz', N=200)
    X_test, y_test = bbefp.dataset.get_data_particlenet('data/test/merged.npz', N=200)

    ckpt_dir = strftime('testckpts_particlenet_%b%d_%H%M%S')

    model = bbefp.particlenet.get_particle_net(
        num_classes=2,
        input_shapes=dict(
            points=(200, 2),
            features=(200, 5),
            mask=(200,1)
            )
        )

    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 10:
            lr *= 0.1
        elif epoch > 20:
            lr *= 0.01
        logging.info('Learning rate: %f'%lr)
        return lr

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
        metrics=['accuracy']
        )
    model.summary()

    print(model)

    model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=20, # --- train only for 1 epoch here for demonstration ---
        validation_data=(X_test, y_test),
        shuffle=True,
        # callbacks=callbacks
        )

if __name__ == '__main__':
    main()