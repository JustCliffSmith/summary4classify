# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
import keras.backend as K
from keras.layers import Input, Embedding, Dense, SpatialDropout1D, LSTM
from keras.layers import RepeatVector, TimeDistributed
import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from pipeline.utility import unpack_data_dict

def build_autoencoder(verbose=False, compile=True):
    """ Build VAE model."""

    MAXLEN = 220 # Is it worth directly inferring this from the training data? Probably.
    path = project_dir + '/data/processed/word_index.pkl'
    with open(path, 'rb') as f:
        word_index = pickle.load(f)
        print(f"Word index length: {len(word_index)}")
    path = project_dir + '/data/processed/embedding_matrix.pkl'
    with open(path, 'rb') as f:
        embedding_matrix = pickle.load(f)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")

    inputs = Input(shape=(MAXLEN,), dtype='int32')
    embedding_layer = Embedding(input_dim=len(word_index) ,
                                output_dim=300,
                                weights=[embedding_matrix],
                                input_length=MAXLEN,
                                trainable=False)

    embedded_text = embedding_layer(inputs)
    #embedded_text = Embedding(,128
    x = LSTM(300)(embedded_text)
    #x = SpatialDropout1D(0.2)(x)
    encoded = RepeatVector(100)(x)
    decoded = LSTM(300, return_sequences=True)(encoded)
    outputs = TimeDistributed(Dense(len(word_index), activation='softmax'))(decoded)
    #x = Dense( , activation='relu')(x)

    model = Model(inputs, outputs)
    if verbose:
        model.summary()
    if compile:
        model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(0.005),
                      metrics=[])
    return model

def train_model(model, data):
    """ """
    n_splits = 5
    BATCH_SIZE = 512
    NUM_EPOCHS = 3

    X_train, y_train, X_test, y_test = unpack_data_dict(data)

    splits = list(KFold(n_splits=n_splits).split(X_train, y_train))

    oof_preds = np.zeros((X_train.shape[0]))
    test_preds = np.zeros((X_test.shape[0]))
    #TODO modify all of this as appropriate for seq2seq_autoencoder 
    for fold in range(n_splits):
        K.clear_session()
        train_index, val_index = splits[fold]
        checkpoint = ModelCheckpoint(f"seq2eq_ae_{fold}.hdf5", 
                                     save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       verbose=1,
                                       patience=3)
        model.fit(X_train[train_index],
                  y_train[train_index]>0.5,
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  validation_data=(X_train[val_index], y_train[val_index]>0.5),
                  callbacks=[early_stopping, checkpoint])
        oof_preds[val_index] += model.predict(X_train[val_index])[:,0]
        test_preds += model.predict(X_test)[:,0]
    test_preds /= 5


def main(project_dir):
    """ 
    """
    logger = logging.getLogger(__name__)

    path = project_dir + '/data/processed/data_tokenize.pkl'
    logging.info(f"Loading tokenized data from {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)    
    autoencoder = build_autoencoder(verbose=True)
    train_model(autoencoder, data)


if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])

    main(project_dir)
