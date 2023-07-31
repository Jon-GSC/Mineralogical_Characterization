'''
 Language: Python 3.8
 Coding: utf-8
 Note: 2. this code is used for model training.
 Acknowledgements: appreciate for sharing the open source code and library
'''
#-----------------------------------------------------------------------------------------------------------------------
import os
import sys
import re
import time
import lasio
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.layers import Input,Flatten,Dense,Reshape
from keras.losses import binary_crossentropy
from tensorflow import keras
import tensorflow as tf
from itools import forWeb_functions as fun
#-----------------------------------------------------------------------------------------------------------------------
savepath0 = os.getcwd()
data_folder = "data/"
model_folder = 'models'
save_folder = 'results'
training_file = 'Training_all.csv'
input_str = ['GR','RD','DTP','DEN','PE']
output_str = ['WQFM', 'WCARB', 'WCLAY', 'WPYR']

n_tile = 8
epochs = 50
batch_size = 32
version = 0
#-----------------------------------------------------------------------------------------------------------------------
t_start = time.time()
input_size = len(input_str) * n_tile
output_size = len(output_str)
basic_name = f'ConvXGB_v{version}'
save_model_name = basic_name + '.model'
modelpath = os.path.join(savepath0,model_folder)
if not os.path.exists(modelpath):
    os.mkdir(modelpath)
savepath = os.path.join(savepath0,save_folder)
if not os.path.exists(savepath):
    os.mkdir(savepath)
#-----------------------------------------------------------------------------------------------------------------------
def main():
    d_slice_all = pd.read_csv(savepath+'/'+training_file)
    d_slice_all = d_slice_all.dropna()  
    d_slice_all = d_slice_all.drop(d_slice_all[d_slice_all['WQFM'] > 1].index)
    d_slice_all = d_slice_all.drop(d_slice_all[d_slice_all['WCARB'] > 1].index)
    d_slice_all = d_slice_all.drop(d_slice_all[d_slice_all['WCLAY'] > 1].index)
    d_slice_all = d_slice_all.drop(d_slice_all[d_slice_all['WPYR'] > 1].index)
    X = d_slice_all[input_str].values
    y = d_slice_all[output_str].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.tile(X, n_tile)
    X_train, X_valid, y_train, y_valid = train_test_split(X.reshape(-1,input_size, 1), y.reshape(-1,output_size, 1), test_size=0.2, random_state=1888)
    for i, val in enumerate(output_str):  
        input_layer0 = Input(shape=(input_size, 1)) 
        model = fun.build_model_1(input_layer0, 1, 32, 3, 0.5)  
        c = keras.optimizers.Adam(learning_rate=0.001) 
        model.compile(loss=fun.rmse, optimizer=c, metrics=[fun.rmse]) 
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1)
        model_checkpoint = ModelCheckpoint(modelpath+'/'+basic_name+f'{i}.model', monitor='val_loss',
                                           mode='min', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.8, patience=5, min_lr=0.00001, verbose=1)
        history = model.fit(X_train, y_train[:,i], validation_data=(X_valid, y_valid[:,i]),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[model_checkpoint, reduce_lr, early_stopping],
                            verbose=2)
        pd.DataFrame(history.history).to_csv(savepath+f'/TrainHistory1_ConvXGB{version}_{i}.csv') 
    print(f'Time used:{(time.time()-t_start)/60} minutes')

if __name__ == '__main__':
    main()
