'''
 Language: Python 3.8+
 Coding: utf-8
 Note: 3. this code is used for predict the mineralogical composition.
 Acknowledgements: appreciate for sharing the open source code and library
'''
# -----------------------------------------------------------------------------------------------------------------------
import os
import sys
import re
import time
import lasio
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras import Model
from itools import forWeb_functions as fun
# -----------------------------------------------------------------------------------------------------------------------
str_list = ['DEPT', 'GR', 'RD', 'DTP', 'DEN', 'PE', 'WQFM', 'WCARB', 'WCLAY', 'WPYR']
GR_index = ['GR', 'GRD', 'SGR', 'GR_EDTC']
RD_index = ['RD', 'RM', 'RS', 'M2RX']
DTP_index = ['DTP', 'DT', 'AC']
DEN_index = ['DEN', 'RHOB']
PE_index = ['PE', 'PEF', 'PEFH']

savepath0 = os.getcwd()
data_folder = "/data/"
model_folder = 'models'
save_folder = 'results'

training_file = 'Training_all.csv'
slst = ['28289.las', ]
slst0 = ['28289.csv', ]
input_str = ['GR', 'RD', 'DTP', 'DEN', 'PE']  #str_list[1:6]
output_str = ['WQFM', 'WCARB', 'WCLAY', 'WPYR']

n_tile = 8
version = 0
# -----------------------------------------------------------------------------------------------------------------------
t_start = time.time()
basic_name = f'ConvXGB_v{version}'
save_model_name = basic_name + '.model'
modelpath = os.path.join(savepath0, model_folder)
if not os.path.exists(modelpath):
    os.mkdir(modelpath)
savepath = os.path.join(savepath0, save_folder)
if not os.path.exists(savepath):
    os.mkdir(savepath)
# -----------------------------------------------------------------------------------------------------------------------
def main():
    d_slice_all0 = pd.read_csv(savepath + '/' + training_file)
    d_slice_all0 = d_slice_all0.dropna()
    d_slice_all0 = d_slice_all0.drop(d_slice_all0[d_slice_all0['WQFM'] > 1].index)
    d_slice_all0 = d_slice_all0.drop(d_slice_all0[d_slice_all0['WCARB'] > 1].index)
    d_slice_all0 = d_slice_all0.drop(d_slice_all0[d_slice_all0['WCLAY'] > 1].index)
    d_slice_all0 = d_slice_all0.drop(d_slice_all0[d_slice_all0['WPYR'] > 1].index)
    X = d_slice_all0[input_str].values
    scaler = StandardScaler().fit(X)
    for i, fn in enumerate(slst):
        print(i, '---', fn)
        d_slice_all = pd.DataFrame(columns=[])
        las = lasio.read(savepath0 + data_folder + fn)
        d_slice0, tops = fun.Las_slice(las)
        d_slice0['DEPT'] = d_slice0.index
        log_list = fun.idenfity_logs(d_slice0, GR_index, RD_index, DTP_index, DEN_index, PE_index)
        if len(log_list) == 10:
            d_slice_all = d_slice0[log_list]
            d_slice_all.rename(columns=dict(zip(log_list, str_list)), inplace=True)
        else:
            print('    Error: filename- ', fn)
            continue
        d_slice_all.insert(loc=0, column='ID', value=fn[:-4])
        d_slice_all.reset_index(drop=True, inplace=True)
        d_slice_all = d_slice_all.dropna()
        d_slice_all = d_slice_all.drop(d_slice_all[d_slice_all['WQFM'] > 1].index)
        d_slice_all = d_slice_all.drop(d_slice_all[d_slice_all['WCARB'] > 1].index)
        d_slice_all = d_slice_all.drop(d_slice_all[d_slice_all['WCLAY'] > 1].index)
        d_slice_all = d_slice_all.drop(d_slice_all[d_slice_all['WPYR'] > 1].index)
        if d_slice_all.empty:  continue
        X2 = d_slice_all[input_str].values
        X2 = scaler.transform(X2)
        X2 = np.tile(X2, n_tile)
        preds_valid = np.empty((X2.shape[0], len(output_str)))
        for i0, val in enumerate(output_str):
            model0 = load_model(modelpath + '/' + basic_name + f'{i0}.model', custom_objects={'rmse': fun.rmse})
            preds_valid[:, i0] = model0.predict(X2).reshape(-1)
        preds_valid = preds_valid / np.sum(preds_valid, axis=1)[:, None]
        d_slice_all['WQFM_pre'] = preds_valid[:, 0]
        d_slice_all['WCARB_pre'] = preds_valid[:, 1]
        d_slice_all['WCLAY_pre'] = preds_valid[:, 2]
        d_slice_all['WPYR_pre'] = preds_valid[:, 3]
        fn_save = os.path.join(savepath, f'{fn[:-4]}.csv')
        d_slice_all.to_csv(fn_save, index=True)
    for i, fn in enumerate(slst0):
        print(i, '---', fn)
        d_slice_all = pd.read_csv(os.path.join(savepath, fn))
        d_slice_all = d_slice_all.dropna()
        mineral_colors = ['yellow', 'lightblue', 'lightcoral', 'violet', ]
        fn_save = os.path.join(savepath, f'p_core_{fn[0:-4]}.png')
        fun.compare_minerals_plot_0(fn_save, d_slice_all, 'Prediction', mineral_colors)
        fun.scatter_plot(savepath, fn, d_slice_all)
        # plt.show();
    print(f'Time used:{(time.time() - t_start) / 60} minutes')

if __name__ == '__main__':
    main()
